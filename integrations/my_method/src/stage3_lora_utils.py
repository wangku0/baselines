from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from .model_utils import load_model_and_processor
from .risk_space.recommended_config import load_recommended_risk_config
from .risk_space.transport_layer_selection import select_layers_by_risk_transport, select_modules_by_flow_rit
from .utils import ensure_dir, logger, resolve_path, save_json


def sync_stage3_layers_with_recommendation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronize Stage 3 layer settings with Stage 1.5 recommendations.

    Stage 1/1.5 layer ids refer to ``outputs.hidden_states`` indices, where
    hidden_states[0] is the embedding output. Therefore, the transformer block
    that writes into hidden_states[L] is block L - 1.
    """
    cfg = copy.deepcopy(config)
    stage3 = cfg.get("stage3", {})
    risk_cfg = stage3.get("risk_space", {})
    if not bool(risk_cfg.get("use_stage1_5_recommended", False)):
        return cfg
    try:
        rec = load_recommended_risk_config(
            cfg,
            allow_fallback=bool(cfg.get("flow_matching", {}).get("recommended_config", {}).get("allow_fallback", False)),
        )
    except Exception as exc:
        logger.warning("Could not synchronize Stage 3 layers from Stage 1.5: %s. Using explicit config layers.", exc)
        return cfg
    cfg.setdefault("stage3", {}).setdefault("risk_space", {})["risk_layers"] = rec.recommended_hidden_layers
    cfg.setdefault("stage3", {}).setdefault("risk_space", {})["k"] = rec.recommended_k
    cfg.setdefault("stage3", {}).setdefault("risk_space", {})["score_mode"] = rec.recommended_score_mode
    cfg.setdefault("stage3", {}).setdefault("risk_space", {})["risk_basis_path"] = rec.risk_basis_path
    selected_hidden_layers = rec.recommended_hidden_layers
    selected_lora_layers = rec.lora_train_layers
    layer_selection = cfg.setdefault("stage3", {}).get("layer_selection", {})
    method = str(layer_selection.get("method", "stage1_5_ablation"))
    if method == "risk_transport_influence":
        selection = select_layers_by_risk_transport(cfg, rec)
        selected_hidden_layers = selection.selected_hidden_layers
        selected_lora_layers = selection.selected_lora_layers
    elif method == "module_risk_transport_influence":
        selection = select_modules_by_flow_rit(cfg, rec)
        selected_hidden_layers = selection.selected_hidden_layers
        selected_lora_layers = selection.selected_lora_layers
        cfg.setdefault("stage3", {}).setdefault("lora", {})["exact_target_modules"] = selection.selected_module_names
    elif method not in {"stage1_5_ablation", "stage1_5_recommended"}:
        raise ValueError(
            f"Unsupported stage3.layer_selection.method={method!r}. "
            "Use 'stage1_5_ablation', 'risk_transport_influence', or 'module_risk_transport_influence'."
        )

    cfg.setdefault("stage3", {}).setdefault("risk_space", {})["risk_layers"] = selected_hidden_layers
    cfg.setdefault("stage3", {}).setdefault("hidden_retain", {})["layers"] = selected_hidden_layers
    cfg.setdefault("stage3", {}).setdefault("lora", {})["train_layers"] = selected_lora_layers

    logger.info(
        "Synchronized Stage 3 layers using %s: risk_layers=%s, hidden_retain.layers=%s, lora.train_layers=%s",
        method,
        selected_hidden_layers,
        selected_hidden_layers,
        selected_lora_layers,
    )
    return cfg


def _stage3_model_config(config: Dict[str, Any], model_path_override: str | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    base = cfg["stage3"]["base_model"]
    cfg.setdefault("model", {})
    cfg["model"]["local_path"] = model_path_override or base.get("model_path") or cfg["model"].get("local_path")
    cfg["model"]["torch_dtype"] = base.get("torch_dtype", cfg["model"].get("torch_dtype", "auto"))
    cfg["model"]["device_map"] = base.get("device_map", cfg["model"].get("device_map", "auto"))
    cfg["model"]["trust_remote_code"] = base.get("trust_remote_code", cfg["model"].get("trust_remote_code", True))
    for key in ("local_files_only", "cache_dir", "max_memory", "offload_folder"):
        if key in base:
            cfg["model"][key] = base[key]
    return cfg


def load_base_model_and_processor(config: Dict[str, Any], model_path_override: str | None = None):
    return load_model_and_processor(_stage3_model_config(config, model_path_override))


def _freeze_base(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _module_has_trainable_device_params(module) -> bool:
    params = list(module.parameters(recurse=False))
    if not params:
        return True
    return all(p.device.type not in {"meta", "cpu"} for p in params)


def _find_exact_lora_modules(model, config: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    lora_cfg = config["stage3"]["lora"]
    suffixes = tuple(lora_cfg.get("target_modules", ["q_proj", "v_proj", "up_proj", "down_proj"]))
    preferred = [int(x) for x in lora_cfg.get("train_layers", [20])]
    fallback = [int(x) for x in lora_cfg.get("fallback_train_nearby_layers", [])]
    require_requested = bool(lora_cfg.get("require_requested_layers", True))
    named = [name for name, _ in model.named_modules()]

    module_map = dict(model.named_modules())
    exact_by_layer: Dict[int, List[str]] = {}
    for layer in preferred:
        exact = []
        pattern = re.compile(rf"(^|\.)(layers|h)\.{layer}\.")
        for name in named:
            if pattern.search(name) and name.endswith(suffixes) and _module_has_trainable_device_params(module_map[name]):
                exact.append(name)
        if exact:
            exact_by_layer[layer] = sorted(exact)

    missing = [layer for layer in preferred if layer not in exact_by_layer]
    if missing and require_requested:
        raise ValueError(
            f"Requested LoRA layer(s) {missing} were not available as trainable non-meta modules. "
            "Refusing to silently train different layers. On 2*T4, set stage3.base_model.max_memory "
            "to include both GPUs, for example {0: '13GiB', 1: '13GiB', cpu: '28GiB'}, "
            "and ensure requested layers are placed on CUDA, not CPU/disk/meta offload."
        )

    if missing:
        for layer in fallback:
            if layer in exact_by_layer or layer in preferred:
                continue
            exact = []
            pattern = re.compile(rf"(^|\.)(layers|h)\.{layer}\.")
            for name in named:
                if pattern.search(name) and name.endswith(suffixes) and _module_has_trainable_device_params(module_map[name]):
                    exact.append(name)
            if exact:
                logger.warning("Requested LoRA layer(s) %s missing; adding nearby layer %s.", missing, layer)
                exact_by_layer[layer] = sorted(exact)
                break

    actual_layers = sorted(exact_by_layer)
    exact_modules = [name for layer in actual_layers for name in exact_by_layer[layer]]
    if not exact_modules:
        raise ValueError(f"Could not find LoRA target modules for requested layers {preferred}.")
    return exact_modules, actual_layers


def _extract_lora_layers_from_names(names: List[str]) -> List[int]:
    layers = set()
    pattern = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
    for name in names:
        match = pattern.search(name)
        if match:
            layers.add(int(match.group(1)))
    return sorted(layers)


def _trainable_lora_module_names(model) -> List[str]:
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            # Convert parameter path back to its parent module path.
            parent = re.sub(r"\.lora_[AB]\.[^.]+\.weight$", "", name)
            parent = parent.replace("base_model.model.", "")
            names.append(parent)
    return sorted(set(names))


def _assert_lora_layers(model, requested_layers: List[int], target_modules: List[str], actual: Dict[str, Any]) -> Dict[str, Any]:
    trainable_modules = _trainable_lora_module_names(model)
    trainable_layers = _extract_lora_layers_from_names(trainable_modules)
    requested_sorted = sorted({int(x) for x in requested_layers})
    actual_layers = sorted({int(x) for x in actual.get("actual_layers", [])})
    if trainable_layers:
        actual_layers = trainable_layers
    expected_suffixes = tuple(target_modules)
    if actual.get("strategy") in {"exact_module_names", "module_risk_transport_exact_module_names"}:
        expected_count = len(actual.get("target_modules", []))
    else:
        expected_count = len(requested_sorted) * len(expected_suffixes)

    if actual_layers != requested_sorted:
        raise RuntimeError(
            "LoRA layer injection mismatch: "
            f"requested_layers={requested_sorted}, actual_layers={actual_layers}, "
            f"trainable_lora_modules={trainable_modules[:20]}"
        )
    if len(trainable_modules) < expected_count:
        raise RuntimeError(
            "LoRA target module count mismatch: "
            f"expected at least {expected_count}, got {len(trainable_modules)}. "
            f"trainable_lora_modules={trainable_modules[:20]}"
        )
    actual["actual_layers"] = actual_layers
    actual["trainable_lora_modules"] = trainable_modules
    return actual


def add_lora_adapter(model, config: Dict[str, Any]):
    _freeze_base(model)
    lora_cfg = config["stage3"]["lora"]
    metrics_dir = ensure_dir(resolve_path(config, config["stage3"]["outputs"]["metrics_dir"]))
    requested_layers = [int(x) for x in lora_cfg.get("train_layers", [20])]
    target_modules = list(lora_cfg.get("target_modules", ["q_proj", "v_proj", "up_proj", "down_proj"]))
    exact_target_modules = [str(x) for x in lora_cfg.get("exact_target_modules", [])]
    if exact_target_modules:
        peft_cfg = LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias=lora_cfg.get("bias", "none"),
            target_modules=exact_target_modules,
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, peft_cfg)
        actual = {
            "strategy": "module_risk_transport_exact_module_names",
            "requested_layers": requested_layers,
            "actual_layers": _extract_lora_layers_from_names(exact_target_modules),
            "target_modules": exact_target_modules,
        }
        actual = _assert_lora_layers(model, requested_layers, target_modules, actual)
        save_json(actual, metrics_dir / "actual_lora_modules.json")
        logger.info("LoRA modules saved to %s", metrics_dir / "actual_lora_modules.json")
        return model
    actual = {"strategy": "layers_to_transform", "requested_layers": requested_layers, "actual_layers": requested_layers, "target_modules": target_modules}
    should_try_layer_api = not hasattr(model, "hf_device_map")
    try:
        if not should_try_layer_api:
            raise RuntimeError("Model was loaded with device_map/offload; using exact non-meta module matching for PEFT training.")
        peft_cfg = LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias=lora_cfg.get("bias", "none"),
            target_modules=target_modules,
            layers_to_transform=requested_layers,
            layers_pattern="layers",
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, peft_cfg)
    except Exception as exc:
        logger.warning("PEFT layers_to_transform LoRA injection failed: %s. Falling back to exact module names.", exc)
        exact_modules, actual_layers = _find_exact_lora_modules(model, config)
        peft_cfg = LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            bias=lora_cfg.get("bias", "none"),
            target_modules=exact_modules,
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, peft_cfg)
        actual = {"strategy": "exact_module_names", "requested_layers": requested_layers, "actual_layers": actual_layers, "target_modules": exact_modules}
    actual = _assert_lora_layers(model, requested_layers, target_modules, actual)
    if sorted(actual["actual_layers"]) != sorted(requested_layers):
        raise RuntimeError(f"LoRA actual_layers must equal requested_layers, got {actual}")
    save_json(actual, metrics_dir / "actual_lora_modules.json")
    logger.info("LoRA modules saved to %s", metrics_dir / "actual_lora_modules.json")
    return model


def add_all_linear_lora_adapter(model, config: Dict[str, Any], *, strategy: str = "all_linear_lora"):
    """Inject LoRA into every linear-module family, matching the TOFU baseline setup."""
    _freeze_base(model)
    lora_cfg = config["stage3"]["lora"]
    module_names = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] != "lm_head"
    }
    if not module_names:
        raise ValueError("Could not find any torch.nn.Linear modules for all-layer LoRA injection.")
    target_modules = sorted(module_names)
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_cfg)
    trainable_modules = _trainable_lora_module_names(model)
    if not trainable_modules:
        raise RuntimeError("All-layer LoRA injection produced no trainable LoRA modules.")
    metrics_dir = ensure_dir(resolve_path(config, config["stage3"]["outputs"]["metrics_dir"]))
    actual = {
        "strategy": strategy,
        "layer_selection": "none",
        "target_module_suffixes": target_modules,
        "trainable_lora_module_count": len(trainable_modules),
        "trainable_lora_modules": trainable_modules,
    }
    save_json(actual, metrics_dir / "actual_lora_modules.json")
    logger.info("All-layer LoRA modules saved to %s", metrics_dir / "actual_lora_modules.json")
    return model


def _safeeraser_projector_modules(model) -> List[str]:
    modules = []
    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"projector", "multi_modal_projector"} or "multi_modal_projector" in name:
            if any(p.requires_grad is not None for p in module.parameters(recurse=True)):
                modules.append(name)
    # PEFT modules_to_save works by matching module-name suffixes. Prefer the
    # top-level projector suffix when available so adapter save/load restores it.
    suffixes = []
    for name in modules:
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"projector", "multi_modal_projector"}:
            suffixes.append(leaf)
    return sorted(set(suffixes or [name.rsplit(".", 1)[-1] for name in modules]))


def _safeeraser_language_lora_modules(model) -> List[str]:
    suffixes = {"up_proj", "k_proj", "down_proj", "v_proj", "q_proj", "o_proj", "gate_proj"}
    modules = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in suffixes and "language_model" in name and _module_has_trainable_device_params(module):
            modules.append(name)
    return sorted(modules)


def add_safeeraser_ga_lora_adapter(model, config: Dict[str, Any]):
    """Inject SafeEraser-style GA LoRA and save the multimodal projector.

    SafeEraser applies LoRA to LLaVA language-model projection/MLP modules
    with r=32, alpha=256, dropout=0.05, and tunes the multimodal projector.
    """
    _freeze_base(model)
    metrics_dir = ensure_dir(resolve_path(config, config["stage3"]["outputs"]["metrics_dir"]))
    target_modules = _safeeraser_language_lora_modules(model)
    if not target_modules:
        raise ValueError(
            "Could not find SafeEraser GA LoRA modules. Expected LLaVA language_model "
            "q/k/v/o/up/down/gate linear modules on a trainable device."
        )
    modules_to_save = _safeeraser_projector_modules(model)
    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
        task_type=config.get("stage3", {}).get("lora", {}).get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_cfg)
    trainable_modules = _trainable_lora_module_names(model)
    if not trainable_modules:
        raise RuntimeError("SafeEraser GA LoRA injection produced no trainable LoRA modules.")
    trainable_projector = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and ("projector" in name or "multi_modal_projector" in name)
    ]
    actual = {
        "strategy": "safeeraser_ga_lora",
        "layer_selection": "none",
        "lora_r": 32,
        "lora_alpha": 256,
        "lora_dropout": 0.05,
        "target_module_suffixes": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
        "target_modules": target_modules,
        "modules_to_save": modules_to_save,
        "tune_mm_projector": True,
        "trainable_lora_module_count": len(trainable_modules),
        "trainable_lora_modules": trainable_modules,
        "trainable_projector_params": trainable_projector,
    }
    save_json(actual, metrics_dir / "actual_lora_modules.json")
    logger.info("SafeEraser GA LoRA modules saved to %s", metrics_dir / "actual_lora_modules.json")
    return model


def get_trainable_parameter_summary(model) -> Dict[str, Any]:
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    ratio = trainable / total if total else 0.0
    return {"trainable_params": trainable, "total_params": total, "trainable_ratio": ratio}


def save_lora_adapter(model, output_dir: str | Path) -> Path:
    path = ensure_dir(Path(output_dir))
    if isinstance(model, PeftModel) or hasattr(model, "save_pretrained"):
        model.save_pretrained(str(path))
    else:
        raise TypeError("Model does not support save_pretrained for LoRA adapter.")
    return path
