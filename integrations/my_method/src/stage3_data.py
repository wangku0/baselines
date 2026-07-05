from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_loader import load_dataset
from .utils import ensure_dir, logger, resolve_path, save_json


def _metrics_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, config["stage3"]["outputs"]["metrics_dir"]))


def _load_stage2_scores(config: Dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config, config["stage3"]["data"]["train_stage2_scores"])
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 2 train scores: {path}. Run scripts/05_stage2_risk_evaluation.py first.")
    return pd.read_csv(path)


def _load_weights(config: Dict[str, Any]) -> Dict[str, float]:
    path = resolve_path(config, config["stage3"]["data"]["train_weights"])
    if not path.exists():
        logger.warning("Missing Stage 2 train weights: %s. Falling back to weight=1.0.", path)
        return {}
    df = pd.read_csv(path)
    return {str(row["sample_id"]): float(row.get("suggested_weight", 1.0)) for _, row in df.iterrows()}


def _score_lookup(scores: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out = {}
    for _, row in scores.iterrows():
        sid = str(row["sample_id"])
        out[sid] = {
            "generated_response": row.get("generated_response"),
            "R_explicit": float(row.get("R_explicit", 0.0)),
            "R_total": float(row.get("R_total", 0.0)),
        }
    return out


def _enrich_sample(sample: Dict[str, Any], scores: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
    item = dict(sample)
    item["sample_id"] = item["id"]
    score = scores.get(item["id"], {})
    value = score.get("generated_response")
    item["response"] = value if isinstance(value, str) and value.strip() else item.get("response") or ""
    item["R_explicit"] = float(score.get("R_explicit", 0.0))
    item["R_total"] = float(score.get("R_total", 0.0))
    item["risk_weight"] = float(weights.get(item["id"], 1.0))
    return item


def _safe_response_mode(config: Dict[str, Any]) -> str:
    mode = str(config["stage3"].get("safe_response", {}).get("mode", "paired_safe_response")).lower()
    aliases = {
        "paired": "paired_safe_response",
        "safe_neighbor": "paired_safe_response",
        "safenb": "paired_safe_response",
        "paired_safe_response": "paired_safe_response",
        "fallback": "fallback_template",
        "refusal": "fallback_template",
        "refusal_template": "fallback_template",
        "fallback_template": "fallback_template",
        "retain": "retain_template",
        "retain_template": "retain_template",
    }
    if mode not in aliases:
        raise ValueError(
            "Unsupported stage3.safe_response.mode: "
            f"{config['stage3'].get('safe_response', {}).get('mode')!r}. "
            "Use paired_safe_response, fallback_template/refusal_template, or retain_template."
        )
    return aliases[mode]


def _select_safe_target(
    *,
    mode: str,
    safe: Dict[str, Any],
    chosen_retains: List[Dict[str, Any]],
    fallback: str,
) -> tuple[str, str, Optional[str]]:
    if mode == "paired_safe_response":
        value = safe.get("response")
        if isinstance(value, str) and value.strip():
            return value, "paired_safe_response", safe.get("sample_id")
        return fallback, "fallback_template", None
    if mode == "retain_template":
        for retain in chosen_retains:
            value = retain.get("response")
            if isinstance(value, str) and value.strip():
                return value, "retain_template", retain.get("sample_id")
        return fallback, "fallback_template", None
    return fallback, "fallback_template", None


def build_stage3_triplets(
    config: Dict[str, Any],
    *,
    max_train_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build harmful/safe/retain triplets for LoRA unlearning.

    Harmful targets never use unsafe responses. The target response can be the
    paired safe-neighbor response, the safety fallback/refusal template, or a
    retain response for ablation.
    """
    stage3 = config["stage3"]
    seed = int(seed if seed is not None else stage3.get("training", {}).get("seed", 42))
    rng = random.Random(seed)
    scores = _load_stage2_scores(config)
    score_map = _score_lookup(scores)
    weights = _load_weights(config)
    samples = load_dataset(config, split="train")
    enriched = [_enrich_sample(s, score_map, weights) for s in samples]

    harmful_by_pair: Dict[str, Dict[str, Any]] = {}
    safe_by_pair: Dict[str, Dict[str, Any]] = {}
    retains: List[Dict[str, Any]] = []
    for sample in enriched:
        stype = sample.get("sample_type")
        pair_id = sample.get("pair_id")
        if stype == "harmful_trigger" and pair_id:
            harmful_by_pair[str(pair_id)] = sample
        elif stype == "safe_neighbor" and pair_id:
            safe_by_pair[str(pair_id)] = sample
        elif stype == "retain":
            retains.append(sample)

    if not retains:
        raise ValueError("No retain samples available for Stage 3.")

    pair_ids = sorted(set(harmful_by_pair) & set(safe_by_pair))
    if not pair_ids:
        raise ValueError("No paired harmful/safe samples found for Stage 3.")
    if len(pair_ids) < len(harmful_by_pair) or len(pair_ids) < len(safe_by_pair):
        logger.warning("Only %d paired harmful/safe records are usable for Stage 3.", len(pair_ids))

    retain_per_pair = int(stage3.get("training", {}).get("retain_per_pair", stage3.get("data", {}).get("retain_ratio", 2)))
    fallback = stage3.get("safe_response", {}).get("fallback_template", "")
    safe_mode = _safe_response_mode(config)
    triplets: List[Dict[str, Any]] = []
    for idx, pair_id in enumerate(pair_ids):
        harmful = dict(harmful_by_pair[pair_id])
        safe = dict(safe_by_pair[pair_id])
        chosen_retains = [retains[(idx * retain_per_pair + j) % len(retains)] for j in range(retain_per_pair)]
        if len(retains) > retain_per_pair:
            chosen_retains = rng.sample(retains, retain_per_pair)
        target, target_source, target_sample_id = _select_safe_target(
            mode=safe_mode,
            safe=safe,
            chosen_retains=chosen_retains,
            fallback=fallback,
        )
        harmful["target_safe_response"] = target
        harmful["target_safe_response_mode"] = safe_mode
        harmful["target_safe_response_source"] = target_source
        harmful["target_safe_response_sample_id"] = target_sample_id
        triplets.append({"harmful": harmful, "safe": safe, "retains": chosen_retains})

    limit = max_train_samples
    if limit is None:
        cfg_limit = stage3.get("data", {}).get("max_train_samples")
        limit = int(cfg_limit) if cfg_limit is not None else None
    if limit is not None:
        triplets = triplets[: int(limit)]

    preview = []
    for triplet in triplets[:5]:
        preview.append(
            {
                "pair_id": triplet["harmful"].get("pair_id"),
                "harmful_sample_id": triplet["harmful"].get("sample_id"),
                "safe_sample_id": triplet["safe"].get("sample_id"),
                "retain_sample_ids": [r.get("sample_id") for r in triplet["retains"]],
                "target_safe_response_mode": triplet["harmful"].get("target_safe_response_mode"),
                "target_safe_response_source": triplet["harmful"].get("target_safe_response_source"),
                "target_safe_response_sample_id": triplet["harmful"].get("target_safe_response_sample_id"),
            }
        )
    save_json(preview, _metrics_dir(config) / "train_triplets_preview.json")
    logger.info("Built %d Stage 3 triplets. Preview saved.", len(triplets))
    return triplets
