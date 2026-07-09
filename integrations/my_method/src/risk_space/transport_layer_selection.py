from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from .recommended_config import RecommendedRiskConfig, hidden_layers_to_lora_layers, load_recommended_risk_config
from ..data_loader import load_dataset
from ..flow_matching.model import FlowVectorField
from ..flow_matching.utils import compute_risk_coefficients, dynamic_implicit_risk_norm, load_dynamic_implicit_normalization
from ..model_utils import infer_input_device, load_model_and_processor, prepare_vl_inputs
from ..utils import ensure_dir, load_json, logger, resolve_path, save_json


@dataclass
class TransportLayerSelection:
    selected_hidden_layers: List[int]
    selected_lora_layers: List[int]
    method: str
    score_mode: str
    k: int
    candidate_layers: List[int]
    layer_scores: List[Dict[str, float]]
    source_risk_basis_path: str
    source_hidden_cache_path: str
    output_path: str
    transport_target: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleRiskTransportSelection:
    selected_hidden_layers: List[int]
    selected_lora_layers: List[int]
    selected_module_names: List[str]
    method: str
    score_mode: str
    k: int
    candidate_layers: List[int]
    candidate_module_suffixes: List[str]
    module_scores: List[Dict[str, Any]]
    source_risk_basis_path: str
    source_flow_teacher_path: str
    output_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_hidden_cache(config: Dict[str, Any], split: str) -> Dict[str, Any]:
    path = resolve_path(config, config.get("outputs", {}).get("hidden_states_dir", "integrations/my_method/outputs/hidden_states")) / f"{split}_hidden_states.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing hidden states cache for transport layer selection: {path}. Run Stage 1 first.")
    data = torch.load(path, map_location="cpu", weights_only=False)
    data["_path"] = str(path)
    return data


def _load_risk_basis(path: str | Path) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing risk basis for transport layer selection: {p}. Run Stage 1.5 first.")
    data = torch.load(p, map_location="cpu", weights_only=False)
    risk_basis = {int(k): v.float() for k, v in data["risk_basis"].items()}
    safe_center = {int(k): v.float() for k, v in data.get("safe_center", {}).items()}
    return risk_basis, safe_center


def _normalize(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    vals = torch.tensor(list(values.values()), dtype=torch.float32)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo + 1e-12:
        return {k: 0.0 for k in values}
    return {k: (float(v) - lo) / (hi - lo) for k, v in values.items()}


def _transport_target_config(config: Dict[str, Any]) -> Dict[str, Any]:
    target_cfg = config.get("stage3", {}).get("layer_selection", {}).get("transport_target", {})
    mode_raw = str(target_cfg.get("mode", "safe_neighbor")).lower()
    aliases = {
        "safe": "safe_neighbor",
        "safenb": "safe_neighbor",
        "safe_nb": "safe_neighbor",
        "safe_neighbor": "safe_neighbor",
        "retain": "retain",
        "mixed": "mixed",
        "mix": "mixed",
        "safe_retain_mix": "mixed",
    }
    mode = aliases.get(mode_raw)
    if mode is None:
        raise ValueError(
            f"Unsupported stage3.layer_selection.transport_target.mode={target_cfg.get('mode')!r}; "
            "use safe_neighbor, retain, or mixed."
        )
    safe_weight = float(target_cfg.get("safe_weight", target_cfg.get("alpha_safe", 0.5)))
    retain_weight = float(target_cfg.get("retain_weight", target_cfg.get("beta_retain", 0.5)))
    if mode == "safe_neighbor":
        safe_weight, retain_weight = 1.0, 0.0
    elif mode == "retain":
        safe_weight, retain_weight = 0.0, 1.0
    else:
        denom = safe_weight + retain_weight
        if denom <= 0:
            raise ValueError("mixed transport target requires safe_weight + retain_weight > 0.")
        safe_weight, retain_weight = safe_weight / denom, retain_weight / denom
    return {"mode": mode, "safe_weight": safe_weight, "retain_weight": retain_weight}


def _choose_retain_indices(metadata: List[Dict[str, Any]], harmful_by_pair: Dict[str, int], pair_ids: List[str], retain_indices: List[int]) -> Dict[str, int]:
    if not retain_indices:
        raise ValueError("Transport target requires retain samples, but none are available in hidden-state metadata.")
    retain_by_sample_index: Dict[Any, List[int]] = defaultdict(list)
    for idx in retain_indices:
        retain_by_sample_index[metadata[idx].get("sample_index")].append(idx)
    out: Dict[str, int] = {}
    for pos, pid in enumerate(pair_ids):
        h_idx = harmful_by_pair[pid]
        same_record = retain_by_sample_index.get(metadata[h_idx].get("sample_index")) or []
        out[pid] = same_record[pos % len(same_record)] if same_record else retain_indices[pos % len(retain_indices)]
    return out


def _stage3_metrics_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, config.get("stage3", {}).get("outputs", {}).get("metrics_dir", "integrations/my_method/outputs/metrics/stage3")))


def _topn_ablation_path(config: Dict[str, Any]) -> Path:
    sel_cfg = config.get("stage3", {}).get("layer_selection", {})
    raw = sel_cfg.get("top_n_ablation_path") or "integrations/my_method/outputs/metrics/stage3/risk_transport_topn_ablation.json"
    return resolve_path(config, raw)


def _resolve_top_n(config: Dict[str, Any], available_layers: Optional[int] = None) -> int:
    sel_cfg = config.get("stage3", {}).get("layer_selection", {})
    raw_top_n = sel_cfg.get("top_n", 1)
    if isinstance(raw_top_n, str) and raw_top_n.lower() == "auto":
        path = _topn_ablation_path(config)
        if not path.exists():
            raise FileNotFoundError(
                f"stage3.layer_selection.top_n=auto but top-n ablation file is missing: {path}. "
                "Run scripts/05_stage2_4_risk_transport_topn_ablation.py first, or pass --layer_selection_top_n <N>."
            )
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        top_n = int(data.get("recommended_top_n", 1))
    else:
        top_n = int(raw_top_n)
    top_n = max(1, top_n)
    if available_layers is not None:
        top_n = min(top_n, max(1, int(available_layers)))
    return top_n


def _risk_coeff(x: torch.Tensor, layer: int, rec: RecommendedRiskConfig, risk_basis: Dict[int, torch.Tensor], safe_center: Dict[int, torch.Tensor]) -> torch.Tensor:
    basis = risk_basis[layer].float()
    k = min(int(rec.recommended_k), basis.shape[0])
    x_used = x.float()
    score_mode = str(rec.recommended_score_mode)
    if score_mode == "centered" or score_mode.startswith("centered_") or score_mode.startswith("paired_delta"):
        if layer not in safe_center:
            raise KeyError(f"safe_center missing for layer {layer}; centered transport selection cannot proceed.")
        x_used = x_used - safe_center[layer].float()
    elif score_mode == "raw" or score_mode.startswith("raw_"):
        pass
    else:
        raise ValueError(f"Unsupported score_mode={rec.recommended_score_mode}")
    return x_used @ basis[:k].T


def select_layers_by_risk_transport(
    config: Dict[str, Any],
    recommended: RecommendedRiskConfig,
) -> TransportLayerSelection:
    """Select edit layers with the document-style transport criterion.

    This is a lightweight layer-level proxy for the document's module-level
    Risk Transport Influence (RTI):

    - RTI: how strongly paired safe->harmful deltas align with the layer's
      average risk-transport direction.
    - risk projection ratio: how much of that delta lives in the learned risk
      subspace.
    - retain overlap: how strongly retain samples project into the same risk
      subspace. Lower is preferred to reduce capability damage.

    The final score prefers high transport influence and low retain overlap.
    """
    stage3 = config.get("stage3", {})
    sel_cfg = stage3.get("layer_selection", {})
    split = str(sel_cfg.get("split", "train"))
    retain_penalty = float(sel_cfg.get("retain_overlap_penalty", 0.5))
    projection_weight = float(sel_cfg.get("risk_projection_weight", 0.5))
    transport_target = _transport_target_config(config)
    min_pairs = int(sel_cfg.get("min_pairs", 1))
    candidates = [int(x) for x in sel_cfg.get("candidate_layers") or config.get("hidden_states", {}).get("target_layers", [])]
    if not candidates:
        candidates = list(recommended.recommended_hidden_layers)
    top_n = _resolve_top_n(config, available_layers=len(candidates))

    hidden = _load_hidden_cache(config, split)
    metadata = hidden["metadata"]
    hidden_by_layer = {int(k): v.float() for k, v in hidden["hidden_states"].items()}
    risk_basis_path = Path(recommended.risk_basis_path)
    if not risk_basis_path.exists():
        fallback = resolve_path(config, config.get("outputs", {}).get("risk_space_dir", "integrations/my_method/outputs/risk_space")) / "risk_basis.pt"
        if fallback.exists():
            logger.warning(
                "Recommended k-specific risk basis not found at %s; falling back to %s.",
                risk_basis_path,
                fallback,
            )
            risk_basis_path = fallback
        else:
            raise FileNotFoundError(
                f"Missing risk basis for transport layer selection: {risk_basis_path}. "
                f"Fallback also missing: {fallback}. Run Stage 1/1.5 first."
            )
    risk_basis, safe_center = _load_risk_basis(risk_basis_path)

    harmful_by_pair: Dict[str, int] = {}
    safe_by_pair: Dict[str, int] = {}
    retain_indices: List[int] = []
    for idx, meta in enumerate(metadata):
        stype = meta.get("sample_type")
        pair_id = meta.get("pair_id")
        if stype == "harmful_trigger" and pair_id:
            harmful_by_pair[str(pair_id)] = idx
        elif stype == "safe_neighbor" and pair_id:
            safe_by_pair[str(pair_id)] = idx
        elif stype == "retain":
            retain_indices.append(idx)

    pair_ids = sorted(set(harmful_by_pair) & set(safe_by_pair))
    if len(pair_ids) < min_pairs:
        raise ValueError(f"Only {len(pair_ids)} paired samples available; min_pairs={min_pairs}.")
    retain_by_pair = (
        _choose_retain_indices(metadata, harmful_by_pair, pair_ids, retain_indices)
        if transport_target["mode"] in {"retain", "mixed"}
        else {}
    )

    raw_rti: Dict[int, float] = {}
    raw_proj_ratio: Dict[int, float] = {}
    raw_retain: Dict[int, float] = {}
    raw_delta_norm: Dict[int, float] = {}
    rows: List[Dict[str, float]] = []

    for layer in candidates:
        if layer not in hidden_by_layer or layer not in risk_basis:
            logger.warning("Skipping transport layer candidate %s because hidden cache or risk basis is missing.", layer)
            continue
        deltas = []
        for pid in pair_ids:
            hi = harmful_by_pair[pid]
            si = safe_by_pair[pid]
            if transport_target["mode"] == "safe_neighbor":
                target_h = hidden_by_layer[layer][si]
            elif transport_target["mode"] == "retain":
                target_h = hidden_by_layer[layer][retain_by_pair[pid]]
            else:
                target_h = (
                    float(transport_target["safe_weight"]) * hidden_by_layer[layer][si]
                    + float(transport_target["retain_weight"]) * hidden_by_layer[layer][retain_by_pair[pid]]
                )
            # Target boundary -> harmful execution.
            deltas.append(hidden_by_layer[layer][hi] - target_h)
        D = torch.stack(deltas, dim=0).float()
        mean_dir = D.mean(dim=0)
        mean_dir_unit = mean_dir / mean_dir.norm().clamp_min(1e-8)
        rti = torch.relu(D @ mean_dir_unit).mean()
        delta_norm = D.norm(dim=1).mean()
        coeff_delta = _risk_coeff(D, layer, RecommendedRiskConfig(
            recommended_k=recommended.recommended_k,
            recommended_score_mode="raw",
            recommended_hidden_layers=recommended.recommended_hidden_layers,
            lora_train_layers=recommended.lora_train_layers,
            risk_basis_path=recommended.risk_basis_path,
            normalization_config=recommended.normalization_config,
            source_path=recommended.source_path,
            recommended_config_path=recommended.recommended_config_path,
        ), risk_basis, safe_center)
        proj_ratio = coeff_delta.norm(dim=1).mean() / delta_norm.clamp_min(1e-8)

        if retain_indices:
            retain_h = hidden_by_layer[layer][retain_indices].float()
            retain_coeff = _risk_coeff(retain_h, layer, recommended, risk_basis, safe_center)
            retain_overlap = retain_coeff.norm(dim=1).mean()
        else:
            retain_overlap = torch.tensor(0.0)

        raw_rti[layer] = float(rti)
        raw_proj_ratio[layer] = float(proj_ratio)
        raw_retain[layer] = float(retain_overlap)
        raw_delta_norm[layer] = float(delta_norm)

    if not raw_rti:
        raise ValueError(f"No valid candidate layers for risk transport selection. candidates={candidates}")

    n_rti = _normalize(raw_rti)
    n_proj = _normalize(raw_proj_ratio)
    n_retain = _normalize(raw_retain)
    for layer in sorted(raw_rti):
        score = n_rti[layer] + projection_weight * n_proj[layer] - retain_penalty * n_retain[layer]
        rows.append(
            {
                "layer": int(layer),
                "score": float(score),
                "risk_transport_influence": raw_rti[layer],
                "risk_projection_ratio": raw_proj_ratio[layer],
                "retain_overlap": raw_retain[layer],
                "delta_norm": raw_delta_norm[layer],
                "normalized_rti": n_rti[layer],
                "normalized_projection": n_proj[layer],
                "normalized_retain_overlap": n_retain[layer],
            }
        )
    rows = sorted(rows, key=lambda r: (r["score"], r["risk_transport_influence"]), reverse=True)
    selected = sorted(int(r["layer"]) for r in rows[:top_n])
    selected_lora = hidden_layers_to_lora_layers(selected)

    out_dir = _stage3_metrics_dir(config)
    out_path = out_dir / "risk_transport_layer_selection.json"
    result = TransportLayerSelection(
        selected_hidden_layers=selected,
        selected_lora_layers=selected_lora,
        method="risk_transport_influence",
        score_mode=recommended.recommended_score_mode,
        k=int(recommended.recommended_k),
        candidate_layers=candidates,
        layer_scores=rows,
        source_risk_basis_path=str(risk_basis_path),
        source_hidden_cache_path=str(hidden["_path"]),
        output_path=str(out_path),
        transport_target=transport_target,
    )
    save_json(result.to_dict(), out_path)
    logger.info(
        "Risk-transport layer selection chose hidden_layers=%s, lora_layers=%s. Details saved to %s",
        selected,
        selected_lora,
        out_path,
    )
    return result


def _module_selection_output_path(config: Dict[str, Any]) -> Path:
    sel_cfg = config.get("stage3", {}).get("layer_selection", {})
    raw = sel_cfg.get("module_selection_path") or "integrations/my_method/outputs/metrics/stage3/module_risk_transport_selection.json"
    return resolve_path(config, raw)


def _resolve_flow_teacher_path(config: Dict[str, Any]) -> Path:
    raw = (
        config.get("stage3", {})
        .get("flow_distillation", {})
        .get("teacher_path")
    )
    if raw:
        return resolve_path(config, raw)
    flow_out = config.get("flow_matching", {}).get("output_dir", "integrations/my_method/outputs/stage2_5_flow")
    return resolve_path(config, f"{flow_out}/flow_teacher.pt")


def _last_input_token_position(attention_mask: torch.Tensor) -> int:
    positions = torch.nonzero(attention_mask[0] > 0, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("attention_mask has no valid input tokens.")
    return int(positions[-1].item())


def _load_flow_teacher_for_selection(config: Dict[str, Any], path: Path, device: torch.device) -> FlowVectorField:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Flow teacher for module-level RIT selection: {path}. "
            "Run Stage 2.5 first, or pass --flow_teacher_path to Stage 3 training."
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    teacher_cfg = data.get("teacher_cfg", {})
    model = FlowVectorField(
        hidden_dim=int(data["hidden_dim"]),
        cond_dim=int(data["cond_dim"]),
        hidden_width=int(teacher_cfg.get("hidden_width", 1024)),
        time_embedding_dim=int(teacher_cfg.get("time_embedding_dim", 128)),
        layer_embedding_dim=int(teacher_cfg.get("layer_embedding_dim", 16)),
        dropout=float(teacher_cfg.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    model._dynamic_conditioning = data.get("dynamic_conditioning") or {}
    model._static_cond_dim = int(data.get("static_cond_dim", int(data["cond_dim"])))
    return model


def _module_selection_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    base = cfg.get("stage3", {}).get("base_model", {})
    if base:
        cfg.setdefault("model", {})
        cfg["model"]["local_path"] = base.get("model_path") or cfg["model"].get("local_path")
        cfg["model"]["torch_dtype"] = base.get("torch_dtype", cfg["model"].get("torch_dtype", "auto"))
        cfg["model"]["device_map"] = base.get("device_map", cfg["model"].get("device_map", "auto"))
        cfg["model"]["trust_remote_code"] = base.get("trust_remote_code", cfg["model"].get("trust_remote_code", True))
        for key in ("local_files_only", "cache_dir", "max_memory", "offload_folder"):
            if key in base:
                cfg["model"][key] = base[key]
    return cfg


def _candidate_module_names(model: torch.nn.Module, config: Dict[str, Any], hidden_layers: List[int]) -> List[Tuple[int, str]]:
    lora_cfg = config.get("stage3", {}).get("lora", {})
    suffixes = tuple(lora_cfg.get("target_modules", ["q_proj", "v_proj", "up_proj", "down_proj"]))
    lora_layers = hidden_layers_to_lora_layers(hidden_layers)
    out: List[Tuple[int, str]] = []
    for name, _module in model.named_modules():
        for layer in lora_layers:
            if f".layers.{layer}." in f".{name}." or f".h.{layer}." in f".{name}.":
                if name.endswith(suffixes):
                    out.append((layer + 1, name))
                break
    return sorted(set(out), key=lambda x: (x[0], x[1]))


def _selection_condition(
    x: torch.Tensor,
    *,
    layer: int,
    rec: RecommendedRiskConfig,
    risk_basis: Dict[int, torch.Tensor],
    safe_center: Dict[int, torch.Tensor],
    explicit_risk: float,
    group_id: int,
) -> torch.Tensor:
    coeff = compute_risk_coefficients(x[None, :], layer, rec, risk_basis, safe_center)[0]
    group = torch.zeros(3, device=x.device, dtype=x.dtype)
    group[int(group_id)] = 1.0
    return torch.cat([x, coeff.to(x.dtype), torch.tensor([float(explicit_risk)], device=x.device, dtype=x.dtype), group], dim=0)[None, :]


def _maybe_append_selection_dynamic_risk(
    cond: torch.Tensor,
    x: torch.Tensor,
    layer: int,
    rec: RecommendedRiskConfig,
    risk_basis: Dict[int, torch.Tensor],
    safe_center: Dict[int, torch.Tensor],
    flow: FlowVectorField,
    lower: Dict[int, float],
    upper: Dict[int, float],
    clip: bool,
) -> torch.Tensor:
    dynamic_cfg = getattr(flow, "_dynamic_conditioning", {}) or {}
    if not bool(dynamic_cfg.get("R_imp_norm_t", False)):
        return cond
    layer_id = torch.tensor([int(layer)], device=x.device)
    r_imp = dynamic_implicit_risk_norm(
        x[None, :],
        layer_id,
        rec,
        risk_basis,
        safe_center,
        lower,
        upper,
        clip=clip,
    )
    return torch.cat([cond, r_imp.to(device=cond.device, dtype=cond.dtype)], dim=-1)


def _module_output_vector(raw_output: Any, last_pos: int) -> Optional[torch.Tensor]:
    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]
    if not torch.is_tensor(raw_output):
        return None
    if raw_output.ndim == 3:
        return raw_output[0, last_pos, :]
    if raw_output.ndim == 2:
        # Linear submodules usually see flattened [tokens, dim] input.
        pos = min(max(int(last_pos), 0), raw_output.shape[0] - 1)
        return raw_output[pos, :]
    if raw_output.ndim == 1:
        return raw_output
    return None


def select_modules_by_flow_rit(
    config: Dict[str, Any],
    recommended: RecommendedRiskConfig,
) -> ModuleRiskTransportSelection:
    """Select exact LoRA modules with Flow-field RIT and retain-overlap.

    This implements the user's original module-level criterion:

        RIT(p) = E_{x in D_H} [ <F_l(h_l(x)), o_p(x)> ]_+
        RO(p)  = E_{x in D_R} || P_{S_l^risk} o_p(x) ||_2
        Score(p) = normalized_RIT(p) * (1 - normalized_RO(p))

    It is only used when stage3.layer_selection.method is explicitly set to
    "module_risk_transport_influence"; the older layer-level selector is left
    untouched.
    """
    stage3 = config.get("stage3", {})
    sel_cfg = stage3.get("layer_selection", {})
    out_path = _module_selection_output_path(config)
    if bool(sel_cfg.get("reuse_module_selection", True)) and out_path.exists():
        data = load_json(out_path)
        return ModuleRiskTransportSelection(
            selected_hidden_layers=[int(x) for x in data.get("selected_hidden_layers", [])],
            selected_lora_layers=[int(x) for x in data.get("selected_lora_layers", [])],
            selected_module_names=[str(x) for x in data.get("selected_module_names", [])],
            method=str(data.get("method", "module_risk_transport_influence")),
            score_mode=str(data.get("score_mode", recommended.recommended_score_mode)),
            k=int(data.get("k", recommended.recommended_k)),
            candidate_layers=[int(x) for x in data.get("candidate_layers", [])],
            candidate_module_suffixes=[str(x) for x in data.get("candidate_module_suffixes", [])],
            module_scores=list(data.get("module_scores", [])),
            source_risk_basis_path=str(data.get("source_risk_basis_path", recommended.risk_basis_path)),
            source_flow_teacher_path=str(data.get("source_flow_teacher_path", _resolve_flow_teacher_path(config))),
            output_path=str(out_path),
        )

    split = str(sel_cfg.get("split", "train"))
    candidate_layers = [int(x) for x in sel_cfg.get("candidate_layers") or config.get("hidden_states", {}).get("target_layers", [])]
    if not candidate_layers:
        candidate_layers = list(recommended.recommended_hidden_layers)
    top_k_modules = int(sel_cfg.get("top_k_modules", sel_cfg.get("top_n_modules", len(candidate_layers))))
    top_k_modules = max(1, top_k_modules)
    max_harmful = sel_cfg.get("module_selection_max_harmful", sel_cfg.get("max_harmful_samples", 64))
    max_retain = sel_cfg.get("module_selection_max_retain", sel_cfg.get("max_retain_samples", 64))
    max_harmful = None if max_harmful is None else int(max_harmful)
    max_retain = None if max_retain is None else int(max_retain)
    max_pixels = sel_cfg.get("max_pixels", config.get("hidden_states", {}).get("max_pixels"))

    risk_basis_path = Path(recommended.risk_basis_path)
    if not risk_basis_path.exists():
        fallback = resolve_path(config, config.get("outputs", {}).get("risk_space_dir", "integrations/my_method/outputs/risk_space")) / "risk_basis.pt"
        risk_basis_path = fallback if fallback.exists() else risk_basis_path
    risk_basis, safe_center = _load_risk_basis(risk_basis_path)

    model, processor = load_model_and_processor(_module_selection_model_config(config))
    device = infer_input_device(model)
    flow_path = _resolve_flow_teacher_path(config)
    flow = _load_flow_teacher_for_selection(config, flow_path, device)
    flow_dtype = next(flow.parameters()).dtype
    if bool((getattr(flow, "_dynamic_conditioning", {}) or {}).get("R_imp_norm_t", False)):
        norm_lower, norm_upper, norm_clip = load_dynamic_implicit_normalization(config)
    else:
        norm_lower, norm_upper, norm_clip = {}, {}, True

    candidates = _candidate_module_names(model, config, candidate_layers)
    if not candidates:
        raise ValueError(
            "No candidate modules found for module-level RIT selection. "
            "Check stage3.layer_selection.candidate_layers and stage3.lora.target_modules."
        )
    module_by_name = dict(model.named_modules())
    hooked_outputs: Dict[str, Any] = {}
    handles = []
    for _layer, name in candidates:
        def _make_hook(n: str):
            def _hook(_module, _inputs, output):
                hooked_outputs[n] = output
            return _hook
        handles.append(module_by_name[name].register_forward_hook(_make_hook(name)))

    raw_rit: Dict[str, float] = {name: 0.0 for _layer, name in candidates}
    raw_ro: Dict[str, float] = {name: 0.0 for _layer, name in candidates}
    counts_h: Dict[str, int] = {name: 0 for _layer, name in candidates}
    counts_r: Dict[str, int] = {name: 0 for _layer, name in candidates}
    skipped: Dict[str, List[str]] = defaultdict(list)

    harmful = load_dataset(config, split=split, sample_types=["harmful_trigger"], max_samples=max_harmful)
    retain = load_dataset(config, split=split, sample_types=["retain"], max_samples=max_retain)
    if not harmful:
        raise ValueError(f"No harmful_trigger samples available for split={split}.")
    if not retain:
        logger.warning("No retain samples available for module-level RO; RO will be zero.")

    try:
        with torch.inference_mode():
            for sample in harmful:
                hooked_outputs.clear()
                inputs = prepare_vl_inputs(processor, sample["image_path"], sample["instruction"], device, max_pixels=max_pixels)
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_pos = _last_input_token_position(inputs["attention_mask"])
                for layer, name in candidates:
                    if layer not in risk_basis or layer >= len(outputs.hidden_states):
                        skipped[name].append("missing_layer_or_risk_basis")
                        continue
                    op = _module_output_vector(hooked_outputs.get(name), last_pos)
                    if op is None:
                        skipped[name].append("missing_module_output")
                        continue
                    h = outputs.hidden_states[layer][0, last_pos, :].detach()
                    if op.numel() != h.numel():
                        skipped[name].append(f"dim_mismatch:{op.numel()}!={h.numel()}")
                        continue
                    cond = _selection_condition(
                        h.float(),
                        layer=layer,
                        rec=recommended,
                        risk_basis=risk_basis,
                        safe_center=safe_center,
                        explicit_risk=1.0,
                        group_id=0,
                    ).to(device=device, dtype=flow_dtype)
                    cond = _maybe_append_selection_dynamic_risk(
                        cond,
                        h.float().to(device=device, dtype=flow_dtype),
                        layer,
                        recommended,
                        risk_basis,
                        safe_center,
                        flow,
                        norm_lower,
                        norm_upper,
                        norm_clip,
                    )
                    t = torch.zeros(1, 1, device=device, dtype=flow_dtype)
                    flow_vec = flow(h[None, :].to(device=device, dtype=flow_dtype), t, cond, torch.tensor([layer], device=device))
                    rit = torch.relu(torch.sum(flow_vec[0].float() * op.detach().float().to(flow_vec.device)))
                    raw_rit[name] += float(rit.detach().cpu())
                    counts_h[name] += 1
                del outputs, inputs

            for sample in retain:
                hooked_outputs.clear()
                inputs = prepare_vl_inputs(processor, sample["image_path"], sample["instruction"], device, max_pixels=max_pixels)
                _ = model(**inputs, output_hidden_states=False, return_dict=True)
                last_pos = _last_input_token_position(inputs["attention_mask"])
                for layer, name in candidates:
                    if layer not in risk_basis:
                        skipped[name].append("missing_risk_basis")
                        continue
                    op = _module_output_vector(hooked_outputs.get(name), last_pos)
                    if op is None:
                        skipped[name].append("missing_module_output")
                        continue
                    basis = risk_basis[layer].to(device=op.device, dtype=op.dtype)
                    k = min(int(recommended.recommended_k), basis.shape[0])
                    if op.numel() != basis.shape[1]:
                        skipped[name].append(f"dim_mismatch:{op.numel()}!={basis.shape[1]}")
                        continue
                    coeff = op.detach().float().to(basis.device) @ basis[:k].float().T
                    raw_ro[name] += float(torch.norm(coeff, p=2).detach().cpu())
                    counts_r[name] += 1
                del inputs
    finally:
        for handle in handles:
            handle.remove()

    raw_rit = {name: (raw_rit[name] / max(counts_h[name], 1)) for _layer, name in candidates}
    raw_ro = {name: (raw_ro[name] / max(counts_r[name], 1)) for _layer, name in candidates}
    n_rit = _normalize({i: raw_rit[name] for i, (_layer, name) in enumerate(candidates)})
    n_ro = _normalize({i: raw_ro[name] for i, (_layer, name) in enumerate(candidates)})
    rows: List[Dict[str, Any]] = []
    for i, (layer, name) in enumerate(candidates):
        score = float(n_rit[i]) * (1.0 - float(n_ro[i]))
        rows.append(
            {
                "hidden_layer": int(layer),
                "lora_layer": int(layer - 1),
                "module_name": name,
                "score": score,
                "rit": float(raw_rit[name]),
                "retain_overlap": float(raw_ro[name]),
                "normalized_rit": float(n_rit[i]),
                "normalized_retain_overlap": float(n_ro[i]),
                "harmful_count": int(counts_h[name]),
                "retain_count": int(counts_r[name]),
                "skipped_reasons_sample": sorted(set(skipped.get(name, [])))[:8],
            }
        )
    rows = sorted(rows, key=lambda r: (r["score"], r["rit"]), reverse=True)
    selected_rows = [r for r in rows if r["harmful_count"] > 0][:top_k_modules]
    if not selected_rows:
        raise ValueError(
            "Module-level RIT selection found no scoreable modules. "
            "This usually means selected target modules have incompatible output dimensions. "
            "Try target_modules with hidden_dim outputs such as q_proj, v_proj, down_proj, o_proj."
        )
    selected_modules = [str(r["module_name"]) for r in selected_rows]
    selected_lora_layers = sorted({int(r["lora_layer"]) for r in selected_rows})
    selected_hidden_layers = sorted({int(r["hidden_layer"]) for r in selected_rows})
    result = ModuleRiskTransportSelection(
        selected_hidden_layers=selected_hidden_layers,
        selected_lora_layers=selected_lora_layers,
        selected_module_names=selected_modules,
        method="module_risk_transport_influence",
        score_mode=recommended.recommended_score_mode,
        k=int(recommended.recommended_k),
        candidate_layers=candidate_layers,
        candidate_module_suffixes=list(config.get("stage3", {}).get("lora", {}).get("target_modules", [])),
        module_scores=rows,
        source_risk_basis_path=str(risk_basis_path),
        source_flow_teacher_path=str(flow_path),
        output_path=str(out_path),
    )
    save_json(result.to_dict(), out_path)
    logger.info(
        "Module-level RIT selection chose hidden_layers=%s, lora_layers=%s, modules=%s. Details saved to %s",
        selected_hidden_layers,
        selected_lora_layers,
        selected_modules,
        out_path,
    )
    return result


def run_risk_transport_topn_ablation(
    config: Dict[str, Any],
    *,
    recommended_config_path: str | None = None,
    max_top_n: int | None = None,
) -> Dict[str, Any]:
    """Ablate how many risk-transport layers should be edited.

    The ablation is intentionally lightweight: it reuses the same per-layer
    risk-transport scores used by Stage 3 and evaluates prefixes of the ranked
    layer list. The recommended top_n is the smallest prefix whose cumulative
    normalized transport score reaches a configurable coverage target, with a
    tie-break on lower retain overlap and fewer edited layers.
    """
    cfg = dict(config)
    stage3 = cfg.setdefault("stage3", {})
    sel_cfg = stage3.setdefault("layer_selection", {})
    old_top_n = sel_cfg.get("top_n", 1)

    rec = load_recommended_risk_config(
        cfg,
        recommended_config_path=recommended_config_path,
        allow_fallback=bool(cfg.get("flow_matching", {}).get("recommended_config", {}).get("allow_fallback", False)),
    )
    candidates = [int(x) for x in sel_cfg.get("candidate_layers") or cfg.get("hidden_states", {}).get("target_layers", [])]
    if not candidates:
        candidates = list(rec.recommended_hidden_layers)
    sweep_max = int(max_top_n or sel_cfg.get("top_n_ablation_max", len(candidates)))
    sweep_max = max(1, min(sweep_max, len(candidates)))

    try:
        sel_cfg["top_n"] = sweep_max
        selection = select_layers_by_risk_transport(cfg, rec)
    finally:
        sel_cfg["top_n"] = old_top_n

    ranked = selection.layer_scores
    ranked = sorted(ranked, key=lambda r: (r["score"], r["risk_transport_influence"]), reverse=True)
    total_positive = sum(max(0.0, float(r.get("score", 0.0))) for r in ranked[:sweep_max])
    if total_positive <= 1e-12:
        total_positive = sum(max(0.0, float(r.get("normalized_rti", 0.0))) for r in ranked[:sweep_max])
    total_positive = max(total_positive, 1e-12)

    coverage_target = float(sel_cfg.get("top_n_coverage_target", 0.85))
    complexity_penalty = float(sel_cfg.get("top_n_complexity_penalty", 0.03))
    retain_penalty = float(sel_cfg.get("top_n_retain_penalty", 0.10))
    marginal_gain_threshold = float(sel_cfg.get("top_n_min_marginal_gain", 0.05))

    rows: List[Dict[str, Any]] = []
    cumulative = 0.0
    prev_coverage = 0.0
    for n in range(1, sweep_max + 1):
        selected_rows = ranked[:n]
        cumulative = sum(max(0.0, float(r.get("score", 0.0))) for r in selected_rows)
        if cumulative <= 1e-12:
            cumulative = sum(max(0.0, float(r.get("normalized_rti", 0.0))) for r in selected_rows)
        coverage = float(cumulative / total_positive)
        marginal_gain = float(coverage - prev_coverage)
        prev_coverage = coverage
        mean_retain = float(sum(float(r.get("normalized_retain_overlap", 0.0)) for r in selected_rows) / n)
        mean_projection = float(sum(float(r.get("normalized_projection", 0.0)) for r in selected_rows) / n)
        mean_rti = float(sum(float(r.get("normalized_rti", 0.0)) for r in selected_rows) / n)
        objective = float(coverage + 0.25 * mean_projection - retain_penalty * mean_retain - complexity_penalty * (n - 1))
        rows.append(
            {
                "top_n": n,
                "hidden_layers": sorted(int(r["layer"]) for r in selected_rows),
                "lora_layers": hidden_layers_to_lora_layers(sorted(int(r["layer"]) for r in selected_rows)),
                "coverage": coverage,
                "marginal_gain": marginal_gain,
                "mean_normalized_rti": mean_rti,
                "mean_normalized_projection": mean_projection,
                "mean_normalized_retain_overlap": mean_retain,
                "objective": objective,
            }
        )

    eligible = [r for r in rows if r["coverage"] >= coverage_target]
    if eligible:
        best = sorted(eligible, key=lambda r: (r["top_n"], r["mean_normalized_retain_overlap"], -r["objective"]))[0]
        reason = (
            f"Smallest top_n reaching coverage_target={coverage_target:.2f}, with retain overlap as tie-breaker; "
            "coverage is measured over positive risk-transport layer scores."
        )
    else:
        best = sorted(rows, key=lambda r: (r["objective"], -r["top_n"]), reverse=True)[0]
        reason = (
            f"No top_n reached coverage_target={coverage_target:.2f}; selected max objective "
            "with complexity and retain-overlap penalties."
        )

    # If adding the next layer gives little marginal coverage, prefer the
    # previous smaller edit set unless it is the first option.
    if best["top_n"] > 1 and best["marginal_gain"] < marginal_gain_threshold:
        prev = rows[best["top_n"] - 2]
        if prev["coverage"] >= max(0.0, coverage_target - marginal_gain_threshold):
            best = prev
            reason += f" Adjusted down because marginal_gain < {marginal_gain_threshold:.2f}."

    out = {
        "method": "risk_transport_topn_ablation",
        "recommended_top_n": int(best["top_n"]),
        "recommended_hidden_layers": best["hidden_layers"],
        "recommended_lora_layers": best["lora_layers"],
        "recommended_k": int(rec.recommended_k),
        "recommended_score_mode": rec.recommended_score_mode,
        "candidate_layers": candidates,
        "max_top_n": sweep_max,
        "coverage_target": coverage_target,
        "complexity_penalty": complexity_penalty,
        "retain_penalty": retain_penalty,
        "marginal_gain_threshold": marginal_gain_threshold,
        "reason": reason,
        "topn_metrics": rows,
        "layer_scores": ranked,
        "source_risk_basis_path": selection.source_risk_basis_path,
        "source_hidden_cache_path": selection.source_hidden_cache_path,
        "transport_target": selection.transport_target,
    }
    out_path = _topn_ablation_path(cfg)
    save_json(out, out_path)
    logger.info(
        "Risk-transport top_n ablation recommended top_n=%s hidden_layers=%s. Saved to %s",
        out["recommended_top_n"],
        out["recommended_hidden_layers"],
        out_path,
    )
    return out
