from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor

from .utils import cuda_oom_help, logger, resolve_path


try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    process_vision_info = None


def ensure_torch_available() -> None:
    if not hasattr(torch, "cuda") or not hasattr(torch, "Tensor"):
        raise ImportError(
            "PyTorch is not fully installed in the current Python environment. "
            "Install dependencies first with: pip install -r requirements.txt"
        )


def parse_torch_dtype(dtype_name: str) -> Any:
    ensure_torch_available()
    dtype_name = str(dtype_name or "auto").lower()
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype={dtype_name}. Use auto, float16, bfloat16, or float32.")
    return mapping[dtype_name]


def _get_model_class(model_type: str, model_name_or_path: str):
    model_type = (model_type or "").lower()
    name = str(model_name_or_path).lower()
    if "llava" in name or "llava" in model_type:
        try:
            from transformers import LlavaForConditionalGeneration

            return LlavaForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Your transformers version does not expose LlavaForConditionalGeneration. "
                "Please upgrade transformers and accelerate."
            ) from exc
    wants_qwen25 = "qwen2.5" in name or "qwen2_5" in model_type or "qwen25" in model_type
    if wants_qwen25:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Your transformers version does not expose Qwen2_5_VLForConditionalGeneration. "
                "Please upgrade transformers to a Qwen2.5-VL compatible release, for example: "
                "pip install -U transformers accelerate qwen-vl-utils"
            ) from exc

    try:
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Your transformers version does not expose Qwen2VLForConditionalGeneration. "
            "Please install/upgrade Qwen-VL compatible dependencies: "
            "pip install -U transformers accelerate qwen-vl-utils"
        ) from exc


def _resolve_pretrained_id(config: Dict[str, Any], path_or_repo_id: str, local_files_only: bool) -> tuple[str, bool]:
    """Return a string suitable for from_pretrained and whether it is local.

    Existing absolute/relative paths are loaded locally. If the path does not
    exist and local_files_only=False, the value is treated as a Hugging Face
    repo id such as Qwen/Qwen2-VL-2B-Instruct. Transformers will download it to
    the local Hugging Face cache and reuse it on later runs.
    """
    raw = str(path_or_repo_id)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = resolve_path(config, raw)
    if candidate.exists():
        return str(candidate), True
    if local_files_only:
        raise FileNotFoundError(
            f"Model path does not exist locally: {candidate}. "
            "Set model.local_files_only=false to allow Hugging Face download, "
            "or pass a valid local path with --model_path."
        )
    if "/" not in raw or raw.startswith("."):
        logger.warning(
            "Model path %s does not exist locally. Because local_files_only=false, "
            "it will be passed to transformers as a remote model id.",
            candidate,
        )
    return raw, False


def infer_input_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model_and_processor(config: Dict[str, Any], model_path_override: Optional[str] = None):
    ensure_torch_available()
    model_cfg = config.get("model", {})
    local_path = model_path_override or model_cfg.get("local_path")
    if not local_path:
        raise ValueError("model.local_path is required. It can be a local path or Hugging Face repo id.")
    is_qwen_model = "qwen" in str(local_path).lower()
    if is_qwen_model and process_vision_info is None:
        raise ImportError(
            "qwen-vl-utils is required for Qwen2-VL/Qwen2.5-VL image preprocessing. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU; Qwen-VL inference will be very slow.")

    dtype = parse_torch_dtype(model_cfg.get("torch_dtype", "auto"))
    device_map = model_cfg.get("device_map", "auto")
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    local_files_only = bool(model_cfg.get("local_files_only", False))
    pretrained_id, is_local = _resolve_pretrained_id(config, local_path, local_files_only)
    source_label = "local path" if is_local else "Hugging Face repo/cache"
    effective_local_files_only = local_files_only or is_local

    common_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": effective_local_files_only,
    }
    if model_cfg.get("cache_dir") is not None:
        cache_dir = resolve_path(config, model_cfg["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        common_kwargs["cache_dir"] = str(cache_dir)

    logger.info("Loading processor from %s: %s", source_label, pretrained_id)
    processor = AutoProcessor.from_pretrained(pretrained_id, **common_kwargs)

    model_cls = _get_model_class(model_cfg.get("model_type", "qwen2_vl"), pretrained_id)
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        **common_kwargs,
    }
    attn_implementation = model_cfg.get("attn_implementation")
    if attn_implementation not in (None, "", "none", "None"):
        kwargs["attn_implementation"] = str(attn_implementation)
    if model_cfg.get("max_memory") is not None:
        visible_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        max_memory = {}
        ignored_devices = []
        for raw_device, limit in model_cfg["max_memory"].items():
            device = int(raw_device) if isinstance(raw_device, str) and raw_device.isdigit() else raw_device
            if isinstance(device, int) and not 0 <= device < visible_cuda_devices:
                ignored_devices.append(device)
                continue
            max_memory[device] = limit
        if ignored_devices:
            logger.warning(
                "Ignoring max_memory entries for unavailable CUDA devices %s; visible logical devices are %s.",
                sorted(set(ignored_devices)),
                list(range(visible_cuda_devices)),
            )
        kwargs["max_memory"] = max_memory
    if model_cfg.get("offload_folder") is not None:
        offload_folder = resolve_path(config, model_cfg["offload_folder"])
        offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(offload_folder)
        kwargs["offload_state_dict"] = True
    if torch.cuda.is_available() and device_map:
        kwargs["device_map"] = device_map
    elif device_map == "auto":
        logger.warning("device_map='auto' requested, but CUDA is unavailable; loading model on CPU.")

    logger.info("Loading model from %s: %s", source_label, pretrained_id)
    try:
        model = model_cls.from_pretrained(pretrained_id, **kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda" in message:
            raise RuntimeError(f"{exc}\n{cuda_oom_help()}") from exc
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {pretrained_id}. If this is Qwen2.5-VL, upgrade transformers. "
            "If you are offline, use a local model path and set local_files_only=true. "
            f"Original error: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        model = model.to(torch.device("cpu"))
    model.eval()
    logger.info("Model loaded. Input device inferred as %s", infer_input_device(model))
    return model, processor


def uses_qwen_vision_utils(processor: Any) -> bool:
    """Use qwen-vl-utils only for Qwen processors, not merely when installed."""
    identities = [
        processor.__class__.__name__,
        processor.__class__.__module__,
        getattr(getattr(processor, "tokenizer", None), "name_or_path", ""),
    ]
    return process_vision_info is not None and any("qwen" in str(value).lower() for value in identities)


def _move_inputs_to_device(inputs: Any, device: Optional[torch.device]) -> Any:
    if device is None:
        return inputs
    try:
        return inputs.to(device)
    except Exception:
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(device)
        return inputs


def prepare_vl_inputs(
    processor,
    image_path: str,
    instruction: str,
    device: Optional[torch.device] = None,
    *,
    max_pixels: Optional[int] = None,
):
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Open once to fail early on corrupt images. Qwen's official utility will
    # reopen/process the path from the chat message below.
    with Image.open(image_path_obj) as img:
        img.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path_obj),
                    **({"max_pixels": int(max_pixels)} if max_pixels is not None else {}),
                },
                {"type": "text", "text": instruction},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if uses_qwen_vision_utils(processor):
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        image = Image.open(image_path_obj).convert("RGB")
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

    return _move_inputs_to_device(inputs, device)
