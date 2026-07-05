"""Local GGUF-backed replacement for the API classifier used by SafeEraser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional


class LocalLlamaGuard:
    """Run SafeEraser's original evaluation prompts with a local LLaMA-Guard.

    This class deliberately knows nothing about the risk formulas used by
    ``my_method``.  It only replaces the remote model call: prompt in, short
    classification label out.
    """

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        repo_id: str = "QuantFactory/Meta-Llama-Guard-2-8B-GGUF",
        filename: str = "Meta-Llama-Guard-2-8B.Q4_K_M.gguf",
        cache_dir: Optional[str] = None,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        max_tokens: int = 16,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama, LlamaGrammar
        except ImportError as exc:
            raise RuntimeError(
                "Local LLaMA-Guard evaluation requires llama-cpp-python. "
                "Install a CUDA-compatible llama-cpp-python wheel in the same "
                "environment used to run eval_all.py."
            ) from exc

        common = {
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }
        if model_path:
            resolved = Path(model_path).expanduser()
            if not resolved.is_file():
                raise FileNotFoundError(f"LLaMA-Guard GGUF not found: {resolved}")
            self.model = Llama(model_path=str(resolved), **common)
        else:
            hub_args = {
                "repo_id": repo_id,
                "filename": filename,
                **common,
            }
            if cache_dir:
                hub_args["cache_dir"] = cache_dir
            self.model = Llama.from_pretrained(**hub_args)

        self.max_tokens = max_tokens
        self._grammar_class = LlamaGrammar
        self._grammars = {}

    @staticmethod
    def _parse_label(raw: str, allowed_labels: Iterable[str]) -> Optional[str]:
        """Return a standalone allowed label, never silently default to safe."""
        text = (raw or "").strip().lower()
        labels = [label.lower() for label in allowed_labels]
        # Prefer the first non-empty line, where classifier models normally put
        # their answer, then fall back to the complete short generation.
        candidates = [text.splitlines()[0] if text else "", text]
        for candidate in candidates:
            matches = []
            for label in labels:
                match = re.search(rf"\b{re.escape(label)}\b", candidate)
                if match:
                    matches.append((match.start(), label))
            if matches:
                return min(matches)[1]
        return None

    def classify(self, prompt: str, allowed_labels: Iterable[str]) -> tuple[Optional[str], str]:
        # [INST] is the model-format wrapper used by the GGUF implementation;
        # the evaluation instructions inside it remain exactly SafeEraser's.
        formatted_prompt = f"[INST] {prompt.strip()} [/INST]"
        labels = tuple(label.lower() for label in allowed_labels)
        if labels not in self._grammars:
            alternatives = " | ".join(f'"{label}"' for label in labels)
            self._grammars[labels] = self._grammar_class.from_string(
                f"root ::= {alternatives}\n"
            )
        result = self.model(
            formatted_prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
            stop=["</s>", "[INST]", "[/INST]", "\n\n"],
            grammar=self._grammars[labels],
        )
        try:
            raw = str(result["choices"][0]["text"]).strip()
        except (KeyError, IndexError, TypeError):
            raw = str(result).strip()
        return self._parse_label(raw, labels), raw
