from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..risk_space.recommended_config import RecommendedRiskConfig, load_recommended_risk_config, save_resolved_recommended_config
from ..risk_space.transport_layer_selection import select_layers_by_risk_transport
from ..utils import ensure_dir, logger, resolve_path, save_json
from .utils import compute_risk_coefficients, compute_risk_delta_coefficients


def _load_risk_basis(path: Path) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    if not path.exists():
        fallback = path.parent.parent / "risk_basis.pt" if path.parent.name.startswith("k_") else path.parent / "risk_basis.pt"
        if fallback.exists():
            logger.warning("Flow matching risk basis not found at %s; falling back to %s.", path, fallback)
            path = fallback
        else:
            raise FileNotFoundError(f"Missing risk basis for flow matching: {path}. Fallback also missing: {fallback}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return (
        {int(k): v.float() for k, v in data["risk_basis"].items()},
        {int(k): v.float() for k, v in data.get("safe_center", {}).items()},
    )


def _load_hidden_cache(config: Dict[str, Any], split: str) -> Dict[str, Any]:
    path = resolve_path(config, config["outputs"]["hidden_states_dir"]) / f"{split}_hidden_states.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing hidden cache: {path}. Run Stage 1 extraction first.")
    return torch.load(path, map_location="cpu", weights_only=False)


def _condition(x: torch.Tensor, coeff: torch.Tensor, r_explicit: float, group_id: int) -> torch.Tensor:
    group = torch.zeros(3, dtype=torch.float32)
    group[int(group_id)] = 1.0
    return torch.cat([x.float(), coeff.float(), torch.tensor([float(r_explicit)], dtype=torch.float32), group], dim=0)


def _flow_target_config(config: Dict[str, Any]) -> Dict[str, Any]:
    target_cfg = config.get("flow_matching", {}).get("target", {})
    mode = str(target_cfg.get("mode", "safe_neighbor")).lower()
    aliases = {
        "safe": "safe_neighbor",
        "safenb": "safe_neighbor",
        "safe_nb": "safe_neighbor",
        "safe_neighbor": "safe_neighbor",
        "retain": "retain",
        "mixed": "mixed",
        "mix": "mixed",
        "alpha_safenb_beta_retain": "mixed",
        "safe_retain_mix": "mixed",
    }
    if mode not in aliases:
        raise ValueError(
            "Unsupported flow_matching.target.mode: "
            f"{target_cfg.get('mode')!r}. Use safe_neighbor, retain, or mixed."
        )
    mode = aliases[mode]
    safe_weight = float(target_cfg.get("safe_weight", target_cfg.get("alpha_safe", 0.5)))
    retain_weight = float(target_cfg.get("retain_weight", target_cfg.get("beta_retain", 0.5)))
    if mode == "safe_neighbor":
        safe_weight, retain_weight = 1.0, 0.0
    elif mode == "retain":
        safe_weight, retain_weight = 0.0, 1.0
    else:
        denom = safe_weight + retain_weight
        if denom <= 0:
            raise ValueError("mixed flow target requires safe_weight + retain_weight > 0.")
        safe_weight, retain_weight = safe_weight / denom, retain_weight / denom
    return {
        "mode": mode,
        "safe_weight": safe_weight,
        "retain_weight": retain_weight,
    }


def _choose_retain_id(
    *,
    harmful_meta: Dict[str, Any],
    retain_ids: List[str],
    retain_by_sample_index: Dict[Any, List[str]],
    pair_index: int,
) -> str:
    same_record = retain_by_sample_index.get(harmful_meta.get("sample_index")) or []
    if same_record:
        return same_record[pair_index % len(same_record)]
    if not retain_ids:
        raise ValueError("Flow target requires retain samples, but none are available in the hidden cache.")
    return retain_ids[pair_index % len(retain_ids)]


def build_flow_features(
    config: Dict[str, Any],
    *,
    split: str = "train",
    max_pairs: Optional[int] = None,
    debug: bool = False,
    recommended_config_path: Optional[str] = None,
) -> Path:
    out_dir = ensure_dir(resolve_path(config, config.get("flow_matching", {}).get("output_dir", "integrations/my_method/outputs/stage2_5_flow")))
    allow_fallback = bool(config.get("flow_matching", {}).get("recommended_config", {}).get("allow_fallback", False)) or debug
    rec = load_recommended_risk_config(config, recommended_config_path=recommended_config_path, allow_fallback=allow_fallback)
    layer_method = str(config.get("stage3", {}).get("layer_selection", {}).get("method", "stage1_5_ablation"))
    if layer_method == "risk_transport_influence":
        selection = select_layers_by_risk_transport(config, rec)
        rec = RecommendedRiskConfig(
            recommended_k=rec.recommended_k,
            recommended_score_mode=rec.recommended_score_mode,
            recommended_hidden_layers=selection.selected_hidden_layers,
            lora_train_layers=selection.selected_lora_layers,
            risk_basis_path=rec.risk_basis_path,
            normalization_config=rec.normalization_config,
            source_path=rec.source_path,
            recommended_config_path=rec.recommended_config_path,
        )
    elif layer_method not in {"stage1_5_ablation", "stage1_5_recommended"}:
        raise ValueError(f"Unsupported layer selection method for flow features: {layer_method}")
    explicit_score_mode = config.get("stage2", {}).get("implicit_risk", {}).get("score_mode")
    if explicit_score_mode and str(explicit_score_mode) != str(rec.recommended_score_mode):
        rec = RecommendedRiskConfig(
            recommended_k=rec.recommended_k,
            recommended_score_mode=str(explicit_score_mode),
            recommended_hidden_layers=rec.recommended_hidden_layers,
            lora_train_layers=rec.lora_train_layers,
            risk_basis_path=rec.risk_basis_path,
            normalization_config=rec.normalization_config,
            source_path=rec.source_path,
            recommended_config_path=rec.recommended_config_path,
        )
    save_resolved_recommended_config(rec, out_dir / "recommended_config_resolved.json")
    representation_cfg = config.get("flow_matching", {}).get("representation", {})
    requested_pooling = representation_cfg.get("pooling", "stage1_last_input_token_cache")
    actual_pooling = "stage1_last_input_token_cache"
    if requested_pooling != actual_pooling:
        raise ValueError(
            "Flow feature extraction is currently using Stage 1 hidden cache representation "
            f"({actual_pooling}), but config requested {requested_pooling!r}. "
            "Set flow_matching.representation.pooling to 'stage1_last_input_token_cache' or implement response-span extraction."
        )
    hidden = _load_hidden_cache(config, split)
    metadata: List[Dict[str, Any]] = hidden["metadata"]
    hidden_by_layer = {int(k): v.float() for k, v in hidden["hidden_states"].items()}
    basis, centers = _load_risk_basis(Path(rec.risk_basis_path))
    metrics_dir = config.get("stage2", {}).get("outputs", {}).get(
        "metrics_dir", "integrations/my_method/outputs/metrics/stage2"
    )
    dynamic_norm_path = resolve_path(config, str(Path(metrics_dir) / "implicit_normalization.json"))
    if not dynamic_norm_path.exists():
        raise FileNotFoundError(
            f"Missing Stage2 implicit normalization for Flow R_imp(t): {dynamic_norm_path}. "
            "Run scripts/05_stage2_risk_evaluation.py before Stage 2.5 Flow training."
        )

    stage2_path = resolve_path(config, config.get("stage3", {}).get("data", {}).get(f"{split}_stage2_scores", f"integrations/my_method/outputs/metrics/stage2/{split}_stage2_risk_scores.csv"))
    explicit = {}
    if stage2_path.exists():
        import pandas as pd

        df = pd.read_csv(stage2_path)
        explicit = {str(r["sample_id"]): float(r.get("R_explicit", 0.0)) for _, r in df.iterrows()}
    else:
        logger.warning("Missing Stage 2 scores for flow conditions: %s. R_explicit will be 0.", stage2_path)

    idx_by_id = {m["id"]: i for i, m in enumerate(metadata)}
    harmful_by_pair: Dict[str, str] = {}
    safe_by_pair: Dict[str, str] = {}
    retain_ids: List[str] = []
    retain_by_sample_index: Dict[Any, List[str]] = defaultdict(list)
    for m in metadata:
        st = m.get("sample_type")
        pid = m.get("pair_id")
        if st == "harmful_trigger" and pid:
            harmful_by_pair[str(pid)] = m["id"]
        elif st == "safe_neighbor" and pid:
            safe_by_pair[str(pid)] = m["id"]
        elif st == "retain":
            retain_ids.append(m["id"])
            retain_by_sample_index[m.get("sample_index")].append(m["id"])
    pair_ids = sorted(set(harmful_by_pair) & set(safe_by_pair))
    if max_pairs is not None:
        pair_ids = pair_ids[: int(max_pairs)]
    if not pair_ids:
        raise ValueError("No paired harmful/safe samples available for flow features.")

    examples = []
    layer_stats = defaultdict(list)
    target_cfg = _flow_target_config(config)
    for pair_index, pid in enumerate(pair_ids):
        hid = harmful_by_pair[pid]
        sid = safe_by_pair[pid]
        rid = _choose_retain_id(
            harmful_meta=metadata[idx_by_id[hid]],
            retain_ids=retain_ids,
            retain_by_sample_index=retain_by_sample_index,
            pair_index=pair_index,
        )
        hi = idx_by_id[hid]
        si = idx_by_id[sid]
        ri = idx_by_id[rid]
        for layer in rec.recommended_hidden_layers:
            xh = hidden_by_layer[layer][hi].float()
            xs = hidden_by_layer[layer][si].float()
            xr = hidden_by_layer[layer][ri].float()
            x_target = target_cfg["safe_weight"] * xs + target_cfg["retain_weight"] * xr
            delta_to_target = xh - x_target
            ch = compute_risk_delta_coefficients(delta_to_target[None, :], layer, rec, basis)[0]
            cs = compute_risk_coefficients(xs[None, :], layer, rec, basis, centers)[0]
            cr = compute_risk_coefficients(xr[None, :], layer, rec, basis, centers)[0]
            ct = compute_risk_coefficients(x_target[None, :], layer, rec, basis, centers)[0]
            examples.append(
                {
                    "kind": "pair",
                    "layer": int(layer),
                    "pair_id": pid,
                    "sample_id": hid,
                    "target_sample_id": sid if target_cfg["mode"] == "safe_neighbor" else rid if target_cfg["mode"] == "retain" else f"{sid}+{rid}",
                    "target_mode": target_cfg["mode"],
                    "target_safe_weight": float(target_cfg["safe_weight"]),
                    "target_retain_weight": float(target_cfg["retain_weight"]),
                    "safe_sample_id": sid,
                    "retain_sample_id": rid,
                    "x0": xh,
                    "x1": x_target,
                    "cond": _condition(xh, ch, explicit.get(hid, 0.0), 0),
                }
            )
            layer_stats[f"delta_norm_layer_{layer}"].append(float(torch.norm(x_target - xh)))
            layer_stats[f"safe_delta_norm_layer_{layer}"].append(float(torch.norm(xs - xh)))
            layer_stats[f"retain_delta_norm_layer_{layer}"].append(float(torch.norm(xr - xh)))
            layer_stats[f"risk_coeff_h_norm_layer_{layer}"].append(float(torch.norm(ch)))
            layer_stats[f"risk_delta_coeff_h_target_norm_layer_{layer}"].append(float(torch.norm(ch)))
            layer_stats[f"risk_coeff_s_norm_layer_{layer}"].append(float(torch.norm(cs)))
            layer_stats[f"risk_coeff_retain_norm_layer_{layer}"].append(float(torch.norm(cr)))
            layer_stats[f"risk_coeff_target_norm_layer_{layer}"].append(float(torch.norm(ct)))

    # Identity examples: same number as pairs where possible.
    identity_source = [safe_by_pair[pid] for pid in pair_ids] + retain_ids[: len(pair_ids)]
    for sid in identity_source:
        if sid not in idx_by_id:
            continue
        si = idx_by_id[sid]
        meta = metadata[si]
        group_id = 1 if meta.get("sample_type") == "safe_neighbor" else 2
        for layer in rec.recommended_hidden_layers:
            x = hidden_by_layer[layer][si].float()
            c = compute_risk_delta_coefficients(torch.zeros_like(x)[None, :], layer, rec, basis)[0]
            examples.append(
                {
                    "kind": "identity",
                    "layer": int(layer),
                    "pair_id": meta.get("pair_id"),
                    "sample_id": sid,
                    "x0": x,
                    "x1": x,
                    "cond": _condition(x, c, explicit.get(sid, 0.0), group_id),
                }
            )

    hidden_dim = int(examples[0]["x0"].numel())
    static_cond_dim = int(examples[0]["cond"].numel())
    cond_dim = static_cond_dim + 1
    feature_path = out_dir / f"features_{split}.pt"
    torch.save(
        {
            "examples": examples,
            "hidden_dim": hidden_dim,
            "cond_dim": cond_dim,
            "static_cond_dim": static_cond_dim,
            "dynamic_conditioning": {
                "R_imp_norm_t": True,
                "normalization": "stage2_sample_risk",
                "normalization_path": str(dynamic_norm_path),
            },
            "recommended": rec.to_dict(),
            "flow_target": target_cfg,
            "representation_pooling": actual_pooling,
            "requested_representation_pooling": requested_pooling,
        },
        feature_path,
    )
    summary = {
        "split": split,
        "num_pairs": len(pair_ids),
        "num_examples": len(examples),
        "recommended_k": rec.recommended_k,
        "recommended_score_mode": rec.recommended_score_mode,
        "recommended_hidden_layers": rec.recommended_hidden_layers,
        "lora_train_layers": rec.lora_train_layers,
        "hidden_dim": hidden_dim,
        "cond_dim": cond_dim,
        "static_cond_dim": static_cond_dim,
        "dynamic_conditioning": {
            "R_imp_norm_t": True,
            "normalization": "stage2_sample_risk",
            "normalization_path": str(dynamic_norm_path),
        },
        "flow_target_mode": target_cfg["mode"],
        "flow_target_safe_weight": target_cfg["safe_weight"],
        "flow_target_retain_weight": target_cfg["retain_weight"],
        "representation_pooling": actual_pooling,
        "requested_representation_pooling": requested_pooling,
        "source_path": rec.source_path,
    }
    for key, values in layer_stats.items():
        if values:
            t = torch.tensor(values)
            summary[key] = {"mean": float(t.mean()), "std": float(t.std(unbiased=False))}
    save_json(summary, out_dir / "feature_summary.json")
    logger.info("Saved flow features to %s", feature_path)
    return feature_path
