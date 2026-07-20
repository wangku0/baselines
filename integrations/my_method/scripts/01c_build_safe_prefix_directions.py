from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_safe_prefix_hidden_states import safe_prefix_hidden_dir
from src.risk_space.recommended_config import load_recommended_risk_config
from src.stage3_lora_utils import sync_stage3_layers_with_recommendation
from src.utils import ensure_dir, load_config, resolve_path


def _load_hidden_cache(config: Dict[str, Any], split: str) -> Dict[str, Any]:
    path = resolve_path(config, config["outputs"]["hidden_states_dir"]) / f"{split}_hidden_states.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing hidden cache: {path}. Run 01_extract_hidden_states.py first.")
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_safe_prefix_cache(config: Dict[str, Any], split: str) -> Dict[str, Any]:
    path = safe_prefix_hidden_dir(config) / f"{split}_safe_prefix_hidden_states.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing safe prefix hidden cache: {path}. "
            "Run 01b_extract_safe_prefix_hidden_states.py first."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_layers(config: Dict[str, Any], hidden_cache: Dict[str, Any], prefix_cache: Dict[str, Any], explicit_layers: Optional[str]) -> List[int]:
    if explicit_layers:
        layers = [int(item.strip()) for item in explicit_layers.split(",") if item.strip()]
    else:
        rec = load_recommended_risk_config(config, allow_fallback=False)
        layers = [int(layer) for layer in rec.recommended_hidden_layers]
    hidden_layers = {int(layer) for layer in hidden_cache["hidden_states"]}
    prefix_layers = {int(layer) for layer in prefix_cache["hidden_states"]}
    missing = [layer for layer in layers if layer not in hidden_layers or layer not in prefix_layers]
    if missing:
        raise KeyError(
            f"Requested layers missing from hidden caches: {missing}. "
            f"hidden_layers={sorted(hidden_layers)}, prefix_layers={sorted(prefix_layers)}"
        )
    return layers


def build_safe_prefix_directions(
    config: Dict[str, Any],
    *,
    split: str = "train",
    layers: Optional[str] = None,
    max_pairs: Optional[int] = None,
    output: Optional[Path] = None,
) -> Path:
    config = sync_stage3_layers_with_recommendation(config)
    hidden = _load_hidden_cache(config, split)
    prefix = _load_safe_prefix_cache(config, split)
    hidden_states = {int(layer): value.float() for layer, value in hidden["hidden_states"].items()}
    prefix_states = {int(layer): value.float() for layer, value in prefix["hidden_states"].items()}
    hidden["hidden_states"] = hidden_states
    prefix["hidden_states"] = prefix_states
    selected_layers = _resolve_layers(config, hidden, prefix, layers)

    hidden_meta = hidden["metadata"]
    prefix_meta = prefix["metadata"]
    harmful_by_pair = {
        str(meta["pair_id"]): idx
        for idx, meta in enumerate(hidden_meta)
        if meta.get("sample_type") == "harmful_trigger" and meta.get("pair_id")
    }
    prefix_by_pair = {
        str(meta["pair_id"]): idx
        for idx, meta in enumerate(prefix_meta)
        if meta.get("sample_type") == "safe_neighbor" and meta.get("pair_id")
    }
    pair_ids = sorted(set(harmful_by_pair) & set(prefix_by_pair))
    if max_pairs is not None:
        pair_ids = pair_ids[: int(max_pairs)]
    if not pair_ids:
        raise ValueError("No overlapping harmful/safe-prefix pair_id values were found.")

    directions: Dict[int, torch.Tensor] = {}
    layer_stats: Dict[str, Dict[str, float]] = {}
    for layer in selected_layers:
        harmful_layer = hidden_states[layer]
        prefix_layer = prefix_states[layer]
        diffs = []
        for pair_id in pair_ids:
            diffs.append(prefix_layer[prefix_by_pair[pair_id]] - harmful_layer[harmful_by_pair[pair_id]])
        stacked = torch.stack(diffs, dim=0)
        direction = stacked.mean(dim=0).float()
        directions[int(layer)] = direction
        norms = stacked.norm(dim=1)
        layer_stats[str(layer)] = {
            "num_pairs": len(pair_ids),
            "mean_pair_delta_norm": float(norms.mean().item()),
            "median_pair_delta_norm": float(norms.median().item()),
            "mean_direction_norm": float(direction.norm().item()),
        }

    out_dir = resolve_path(
        config,
        config.get("flow_matching", {}).get("output_dir", "integrations/my_method/outputs/stage2_5_flow"),
    )
    if output is None:
        output = out_dir / f"safe_prefix_direction_{split}.pt"
    elif not output.is_absolute():
        output = resolve_path(config, str(output))
    ensure_dir(output.parent)
    payload = {
        "directions": directions,
        "layers": selected_layers,
        "split": split,
        "num_pairs": len(pair_ids),
        "pair_ids": pair_ids,
        "source_hidden_states": str(resolve_path(config, config["outputs"]["hidden_states_dir"]) / f"{split}_hidden_states.pt"),
        "source_safe_prefix_hidden_states": str(safe_prefix_hidden_dir(config) / f"{split}_safe_prefix_hidden_states.pt"),
        "safe_answer_prefix_tokens": prefix.get("safe_answer_prefix_tokens"),
        "stats": layer_stats,
    }
    torch.save(payload, output)
    summary = {key: value for key, value in payload.items() if key not in {"directions", "pair_ids"}}
    summary["output"] = str(output)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mean safe-answer-prefix decode steering directions.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--layers", default=None, help="Optional comma-separated hidden layers. Defaults to Stage1.5 recommended layers.")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    build_safe_prefix_directions(
        load_config(args.config),
        split=args.split,
        layers=args.layers,
        max_pairs=args.max_pairs,
        output=args.output,
    )


if __name__ == "__main__":
    main()
