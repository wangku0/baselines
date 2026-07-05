from pathlib import Path
import json
import pandas as pd


ROOT = Path(".").resolve()


def project_root():
    if (ROOT / "outputs").exists() and (ROOT / "configs").exists():
        return ROOT
    if (ROOT / "my_method").exists():
        return ROOT / "my_method"
    return ROOT


PROJ = project_root()


def p(rel):
    return PROJ / rel


def load_json(rel):
    path = p(rel)
    if not path.exists():
        print(f"[MISSING] {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(rel):
    path = p(rel)
    if not path.exists():
        print(f"[MISSING] {path}")
        return None
    return pd.read_csv(path)


def fmt(x, ndigits=4):
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def print_title(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def file_status(rel):
    path = p(rel)
    status = "OK" if path.exists() else "MISSING"
    size = path.stat().st_size if path.exists() else 0
    print(f"{status:8s} {rel}  ({size} bytes)")


def balanced_clearance(clearance):
    value = clearance.get("balanced_explicit_implicit_clearance")
    if value is not None:
        return value
    explicit = clearance.get("explicit_clearance")
    implicit = clearance.get("implicit_clearance")
    if explicit is None or implicit is None:
        return None
    try:
        return 0.5 * float(explicit) + 0.5 * float(implicit)
    except Exception:
        return None


def fusion_clearance(clearance):
    return clearance.get("fusion_total_risk_clearance", clearance.get("total_clearance"))


def metric_pair(item, name):
    before = item.get(f"{name}_before_mean", item.get(f"{name}_mean_before"))
    after = item.get(f"{name}_after_mean", item.get(f"{name}_mean_after"))
    return before, after


def show_output_files():
    print_title("Key Output File Status")

    paths = [
        "integrations/my_method/outputs/hidden_states/train_hidden_states.pt",
        "integrations/my_method/outputs/hidden_states/val_hidden_states.pt",
        "integrations/my_method/outputs/risk_space/risk_basis.pt",
        "integrations/my_method/outputs/metrics/train_risk_space_eval_raw.json",
        "integrations/my_method/outputs/metrics/val_risk_space_eval_raw.json",
        "integrations/my_method/outputs/metrics/stage1_5/recommended_config.json",
        "integrations/my_method/outputs/metrics/stage1_5/recommended_layers.json",
        "integrations/my_method/outputs/metrics/stage1_5/k_sweep_summary.csv",
        "integrations/my_method/outputs/metrics/stage2/train_stage2_summary.json",
        "integrations/my_method/outputs/metrics/stage2/val_stage2_summary.json",
        "integrations/my_method/outputs/metrics/stage2/train_stage2_risk_scores.csv",
        "integrations/my_method/outputs/metrics/stage2/val_stage2_risk_scores.csv",
        "integrations/my_method/outputs/metrics/stage2/train_training_weights.csv",
        "integrations/my_method/outputs/metrics/stage2/implicit_normalization.json",
        "integrations/my_method/outputs/stage2_5_flow/features_train.pt",
        "integrations/my_method/outputs/stage2_5_flow/feature_summary.json",
        "integrations/my_method/outputs/stage2_5_flow/recommended_config_resolved.json",
        "integrations/my_method/outputs/stage2_5_flow/train_log.csv",
        "integrations/my_method/outputs/stage2_5_flow/flow_teacher.pt",
        "integrations/my_method/outputs/stage2_5_flow/flow_config_resolved.yaml",
        "integrations/my_method/outputs/stage2_5_flow/eval_summary.json",
        "integrations/my_method/outputs/stage3/lora_unlearned/adapter/adapter_config.json",
        "integrations/my_method/outputs/stage3/lora_flow_unlearned/adapter/adapter_config.json",
        "integrations/my_method/outputs/metrics/stage3/actual_lora_modules.json",
        "integrations/my_method/outputs/metrics/stage3/risk_transport_topn_ablation.json",
        "integrations/my_method/outputs/metrics/stage3/train_loss_log.csv",
        "integrations/my_method/outputs/metrics/stage3/recommended_config_resolved.json",
        "integrations/my_method/outputs/metrics/stage3/val_unlearned_summary.json",
        "integrations/my_method/outputs/metrics/stage3/val_before_after_summary.json",
        "integrations/my_method/outputs/metrics/stage3/val_unlearned_risk_scores.csv",
    ]

    for rel in paths:
        file_status(rel)


def show_stage1():
    print_title("Stage 1: Risk Space Validation Results")

    for split in ["train", "val"]:
        data = load_json(f"integrations/my_method/outputs/metrics/{split}_risk_space_eval_raw.json")
        if data is None:
            continue

        stats = data.get("stats_by_sample_type", {})
        auc = data.get("auc", {})
        paired = data.get("paired_analysis", {})

        print(f"\n[{split}]")
        for stype in ["harmful_trigger", "safe_neighbor", "retain"]:
            item = stats.get(stype, {})
            print(
                f"  {stype:16s} "
                f"count={item.get('count', 'NA')}, "
                f"mean={fmt(item.get('mean'))}, "
                f"std={fmt(item.get('std'))}, "
                f"median={fmt(item.get('median'))}"
            )

        print(f"  harmful vs safe AUC: {fmt(auc.get('harmful_trigger_vs_safe_neighbor'))}")
        print(f"  harmful vs retain AUC: {fmt(auc.get('harmful_trigger_vs_retain'))}")
        print(f"  paired mean diff: {fmt(paired.get('mean_diff'))}")
        print(f"  harmful > safe ratio: {fmt(paired.get('ratio_harmful_greater_than_safe'))}")


def show_stage1_5():
    print_title("Stage 1.5: Risk Space Calibration Results")

    rec_cfg = load_json("integrations/my_method/outputs/metrics/stage1_5/recommended_config.json")
    rec_layers = load_json("integrations/my_method/outputs/metrics/stage1_5/recommended_layers.json")

    if rec_cfg:
        print("\n[Recommended k / score_mode]")
        print(f"  recommended_k: {rec_cfg.get('recommended_k')}")
        print(f"  recommended_score_mode: {rec_cfg.get('recommended_score_mode')}")
        print(f"  reason: {rec_cfg.get('reason')}")

        val = rec_cfg.get("val_metrics", {})
        train = rec_cfg.get("train_metrics", {})

        print("\n  val metrics:")
        print(f"    harmful_mean: {fmt(val.get('harmful_mean'))}")
        print(f"    safe_mean: {fmt(val.get('safe_mean'))}")
        print(f"    retain_mean: {fmt(val.get('retain_mean'))}")
        print(f"    harmful_minus_safe_mean: {fmt(val.get('harmful_minus_safe_mean'))}")
        print(f"    harmful_minus_retain_mean: {fmt(val.get('harmful_minus_retain_mean'))}")
        print(f"    harmful_vs_safe_auc: {fmt(val.get('harmful_vs_safe_auc'))}")
        print(f"    paired_mean_diff: {fmt(val.get('paired_mean_diff'))}")
        print(f"    harmful > safe ratio: {fmt(val.get('paired_ratio_harmful_greater_than_safe'))}")

        print("\n  train metrics:")
        print(f"    harmful_vs_safe_auc: {fmt(train.get('harmful_vs_safe_auc'))}")
        print(f"    paired_mean_diff: {fmt(train.get('paired_mean_diff'))}")

    if rec_layers:
        print("\n[Recommended layers]")
        print(f"  recommended_layers: {rec_layers.get('recommended_layers')}")
        print(f"  recommended_k: {rec_layers.get('recommended_k')}")
        print(f"  recommended_score_mode: {rec_layers.get('recommended_score_mode')}")

        layers = rec_layers.get("recommended_layers") or []
        try:
            lora_layers = sorted({int(x) - 1 for x in layers})
            print(f"  auto_lora_train_layers: {lora_layers}")
        except Exception:
            print("  auto_lora_train_layers: NA")

        val = rec_layers.get("val_metrics", {})
        print("\n  val layer metrics:")
        print(f"    layer_set_name: {val.get('layer_set_name')}")
        print(f"    layers: {val.get('layers')}")
        print(f"    harmful_mean: {fmt(val.get('harmful_mean'))}")
        print(f"    safe_mean: {fmt(val.get('safe_mean'))}")
        print(f"    retain_mean: {fmt(val.get('retain_mean'))}")
        print(f"    harmful_vs_safe_auc: {fmt(val.get('harmful_vs_safe_auc'))}")
        print(f"    paired_mean_diff: {fmt(val.get('paired_mean_diff'))}")
        print(f"    harmful > safe ratio: {fmt(val.get('paired_ratio_harmful_greater_than_safe'))}")

    sweep = load_csv("integrations/my_method/outputs/metrics/stage1_5/k_sweep_summary.csv")
    if sweep is not None:
        print("\n[k sweep summary: val split]")
        val_sweep = sweep[sweep["split"] == "val"].copy()
        cols = [
            "k",
            "score_mode",
            "harmful_mean",
            "safe_mean",
            "retain_mean",
            "harmful_vs_safe_auc",
            "paired_mean_diff",
            "paired_ratio_harmful_greater_than_safe",
        ]
        cols = [c for c in cols if c in val_sweep.columns]
        print(val_sweep[cols].to_string(index=False))


def show_stage2():
    print_title("Stage 2: Explicit, Implicit, and Total Risk Results")

    norm = load_json("integrations/my_method/outputs/metrics/stage2/implicit_normalization.json")
    if norm:
        print("\n[Implicit Risk Normalization Parameters]")
        print(f"  method: {norm.get('method')}")
        print(f"  lower_percentile: {norm.get('lower_percentile')}")
        print(f"  upper_percentile: {norm.get('upper_percentile')}")
        print(f"  lower_value: {fmt(norm.get('lower_value'))}")
        print(f"  upper_value: {fmt(norm.get('upper_value'))}")
        print(f"  train_count: {norm.get('train_count')}")
        settings = norm.get("implicit_settings", {})
        print(f"  implicit k: {settings.get('k')}")
        print(f"  implicit score_mode: {settings.get('score_mode')}")
        print(f"  implicit layers: {settings.get('layers')}")

    for split in ["train", "val"]:
        summary = load_json(f"integrations/my_method/outputs/metrics/stage2/{split}_stage2_summary.json")
        if summary is None:
            continue

        print(f"\n[{split}]")
        by_type = summary.get("by_sample_type", {})
        for stype in ["harmful_trigger", "safe_neighbor", "retain"]:
            item = by_type.get(stype, {})
            print(f"  {stype}:")
            print(f"    count: {item.get('count', 'NA')}")
            print(f"    R_explicit_mean: {fmt(item.get('R_explicit_mean'))}")
            print(f"    R_implicit_norm_mean: {fmt(item.get('R_implicit_norm_mean'))}")
            print(f"    R_total_mean: {fmt(item.get('R_total_mean'))}")
            print(f"    refusal_rate: {fmt(item.get('refusal_rate'))}")
            print(f"    high_explicit_risk_rate: {fmt(item.get('high_explicit_risk_rate'))}")

        hs = summary.get("harmful_vs_safe", {})
        print("\n  harmful vs safe:")
        print(f"    explicit_auc: {fmt(hs.get('explicit_auc'))}")
        print(f"    implicit_auc: {fmt(hs.get('implicit_auc'))}")
        print(f"    total_risk_auc: {fmt(hs.get('total_risk_auc'))}")
        print(f"    paired_total_risk_mean_diff: {fmt(hs.get('paired_total_risk_mean_diff'))}")
        print(f"    harmful total > safe ratio: {fmt(hs.get('ratio_harmful_total_risk_greater_than_safe'))}")

        hr = summary.get("harmful_vs_retain", {})
        print("\n  harmful vs retain:")
        print(f"    explicit_auc: {fmt(hr.get('explicit_auc'))}")
        print(f"    implicit_auc: {fmt(hr.get('implicit_auc'))}")
        print(f"    total_risk_auc: {fmt(hr.get('total_risk_auc'))}")

        scores = load_csv(f"integrations/my_method/outputs/metrics/stage2/{split}_stage2_risk_scores.csv")
        if scores is not None and "R_implicit_norm_before_clip" in scores.columns:
            print("\n  implicit norm saturation:")
            for stype, g in scores.groupby("sample_type"):
                zero_rate = (g["R_implicit_norm"] == 0).mean()
                one_rate = (g["R_implicit_norm"] == 1).mean()
                print(f"    {stype}: zero_rate={fmt(zero_rate)}, one_rate={fmt(one_rate)}")


def show_stage2_5():
    print_title("Stage 2.5: Flow Matching Teacher Results")

    rec = load_json("integrations/my_method/outputs/stage2_5_flow/recommended_config_resolved.json")
    if rec:
        print("\n[Resolved Recommended Config]")
        print(f"  recommended_k: {rec.get('recommended_k')}")
        print(f"  recommended_score_mode: {rec.get('recommended_score_mode')}")
        print(f"  recommended_hidden_layers: {rec.get('recommended_hidden_layers')}")
        print(f"  lora_train_layers: {rec.get('lora_train_layers')}")
        print(f"  risk_basis_path: {rec.get('risk_basis_path')}")
        print(f"  source_path: {rec.get('source_path')}")

    summary = load_json("integrations/my_method/outputs/stage2_5_flow/feature_summary.json")
    if summary:
        print("\n[Feature Summary]")
        keys = [
            "split",
            "num_pairs",
            "num_examples",
            "recommended_k",
            "recommended_score_mode",
            "recommended_hidden_layers",
            "lora_train_layers",
            "flow_target_mode",
            "flow_target_safe_weight",
            "flow_target_retain_weight",
            "hidden_dim",
            "cond_dim",
            "representation_pooling",
        ]
        for k in keys:
            print(f"  {k}: {summary.get(k)}")

        stat_keys = [k for k in summary.keys() if "norm" in k and isinstance(summary.get(k), dict)]
        if stat_keys:
            print("\n  norm stats:")
            for k in stat_keys:
                v = summary[k]
                print(f"    {k}: mean={fmt(v.get('mean'))}, std={fmt(v.get('std'))}")

    train_log = load_csv("integrations/my_method/outputs/stage2_5_flow/train_log.csv")
    if train_log is not None:
        print("\n[Flow Teacher Train Log]")
        print(f"  steps logged: {len(train_log)}")
        cols = ["step", "loss", "loss_velocity", "loss_endpoint", "loss_identity", "endpoint_noop", "delta_cos"]
        cols = [c for c in cols if c in train_log.columns]
        print(train_log[cols].tail(10).to_string(index=False))

        if "loss_endpoint" in train_log.columns and "endpoint_noop" in train_log.columns:
            last = train_log.iloc[-1]
            endpoint = float(last["loss_endpoint"])
            noop = float(last["endpoint_noop"])
            improvement = 1.0 - endpoint / max(noop, 1e-8)
            print(f"\n  endpoint improvement ratio(last): {fmt(improvement)}")

    eval_summary = load_json("integrations/my_method/outputs/stage2_5_flow/eval_summary.json")
    if eval_summary:
        print("\n[Flow Eval Summary]")
        for k, v in eval_summary.items():
            if isinstance(v, (str, int, float)):
                print(f"  {k}: {v}")


def show_stage3():
    print_title("Stage 3: LoRA / Flow-guided LoRA Unlearning Results")

    topn = load_json("integrations/my_method/outputs/metrics/stage3/risk_transport_topn_ablation.json")
    if topn:
        print("\n[Risk-transport top_n Ablation Recommendation]")
        print(f"  recommended_top_n: {topn.get('recommended_top_n')}")
        print(f"  recommended_hidden_layers: {topn.get('recommended_hidden_layers')}")
        print(f"  recommended_lora_layers: {topn.get('recommended_lora_layers')}")
        print(f"  coverage_target: {topn.get('coverage_target')}")
        print(f"  reason: {topn.get('reason')}")
        rows = topn.get("topn_metrics") or []
        if rows:
            df = pd.DataFrame(rows)
            cols = [
                "top_n",
                "hidden_layers",
                "lora_layers",
                "coverage",
                "marginal_gain",
                "mean_normalized_retain_overlap",
                "objective",
            ]
            cols = [c for c in cols if c in df.columns]
            print("\n  top_n metrics:")
            print(df[cols].to_string(index=False))

    rec = load_json("integrations/my_method/outputs/metrics/stage3/recommended_config_resolved.json")
    if rec:
        print("\n[Stage 3 Resolved Recommended Config]")
        print(f"  recommended_k: {rec.get('recommended_k')}")
        print(f"  recommended_score_mode: {rec.get('recommended_score_mode')}")
        print(f"  recommended_hidden_layers: {rec.get('recommended_hidden_layers')}")
        print(f"  lora_train_layers: {rec.get('lora_train_layers')}")

    actual = load_json("integrations/my_method/outputs/metrics/stage3/actual_lora_modules.json")
    if actual:
        print("\n[Actual LoRA Injected Modules]")
        print(f"  strategy: {actual.get('strategy')}")
        print(f"  requested_layers: {actual.get('requested_layers')}")
        print(f"  actual_layers: {actual.get('actual_layers')}")
        print("  target_modules:")
        for m in actual.get("target_modules", []):
            print(f"    - {m}")

    train_log = load_csv("integrations/my_method/outputs/metrics/stage3/train_loss_log.csv")
    if train_log is not None:
        print("\n[Training Loss Log]")
        print(f"  steps: {len(train_log)}")
        cols = [
            "step",
            "loss_total",
            "loss_safe_ce",
            "loss_align",
            "loss_implicit",
            "loss_safe_kl",
            "loss_retain_kl",
            "loss_retain_hidden",
            "loss_flow_distill",
            "loss_flow_delta",
            "loss_flow_cos",
            "loss_flow_risk",
            "loss_flow_identity",
            "lambda_flow",
            "harmful_delta_lora_norm",
            "harmful_delta_flow_norm",
            "harmful_delta_cos_lora_flow",
            "safe_delta_norm",
            "retain_delta_norm",
        ]
        cols = [c for c in cols if c in train_log.columns]
        print(train_log[cols].tail(10).to_string(index=False))

    summary = load_json("integrations/my_method/outputs/metrics/stage3/val_unlearned_summary.json")
    if summary:
        print("\n[val unlearned summary]")
        by_type = summary.get("by_sample_type", {})
        for stype in ["harmful_trigger", "safe_neighbor", "retain"]:
            item = by_type.get(stype, {})
            if not item:
                continue
            print(f"  {stype}:")
            print(f"    count: {item.get('count', 'NA')}")
            print(f"    R_explicit_mean: {fmt(item.get('R_explicit_mean'))}")
            print(f"    R_implicit_raw_mean: {fmt(item.get('R_implicit_raw_mean'))}")
            print(f"    R_implicit_norm_mean: {fmt(item.get('R_implicit_norm_mean'))}")
            print(f"    R_total_mean: {fmt(item.get('R_total_mean'))}")
            print(f"    R_implicit_norm_zero_rate: {fmt(item.get('R_implicit_norm_zero_rate'))}")
            print(f"    norm_saturation_rate: {fmt(item.get('norm_saturation_rate'))}")
            print(f"    high_risk_rate: {fmt(item.get('high_risk_rate'))}")
            print(f"    refusal_rate: {fmt(item.get('refusal_rate'))}")

    scores = load_csv("integrations/my_method/outputs/metrics/stage3/val_unlearned_risk_scores.csv")
    if scores is not None:
        print("\n[val unlearned risk score raw/norm]")
        cols = ["R_implicit_raw", "R_implicit_norm_before_clip", "R_implicit_norm", "R_total"]
        cols = [c for c in cols if c in scores.columns]
        if cols:
            print(scores.groupby("sample_type")[cols].describe().to_string())

        meta_cols = [
            "recommended_k",
            "recommended_score_mode",
            "recommended_layers",
            "flow_delta_norm",
            "lora_delta_norm",
            "cos_delta_lora_flow",
        ]
        meta_cols = [c for c in meta_cols if c in scores.columns]
        if meta_cols:
            print("\n  evaluation metadata columns sample:")
            print(scores[meta_cols].head(5).to_string(index=False))

    comp = load_json("integrations/my_method/outputs/metrics/stage3/val_before_after_summary.json")
    if comp:
        print("\n[val before/after comparison]")
        print(f"  baseline_response_source: {comp.get('baseline_response_source')}")
        print(f"  unlearned_response_source: {comp.get('unlearned_response_source')}")

        by_type = comp.get("by_sample_type", {})
        for stype in ["harmful_trigger", "safe_neighbor", "retain"]:
            item = by_type.get(stype)
            if not item:
                continue
            print(f"\n  {stype}:")
            print(f"    count: {item.get('count')}")
            before, after = metric_pair(item, "R_explicit")
            print(f"    R_explicit before -> after: {fmt(before)} -> {fmt(after)}")
            before, after = metric_pair(item, "R_implicit_norm")
            print(f"    R_implicit_norm before -> after: {fmt(before)} -> {fmt(after)}")
            before, after = metric_pair(item, "R_implicit_raw")
            if before is not None or after is not None:
                print(f"    R_implicit_raw before -> after: {fmt(before)} -> {fmt(after)}")
            before, after = metric_pair(item, "R_total")
            print(f"    R_total before -> after: {fmt(before)} -> {fmt(after)}")
            print(f"    high_risk_rate before -> after: {fmt(item.get('high_risk_rate_before'))} -> {fmt(item.get('high_risk_rate_after'))}")
            print(f"    refusal_rate before -> after: {fmt(item.get('refusal_rate_before'))} -> {fmt(item.get('refusal_rate_after'))}")

        clearance = comp.get("clearance", {})
        if clearance:
            print("\n  harmful clearance:")
            print(f"    explicit_clearance: {fmt(clearance.get('explicit_clearance'))}")
            print(f"    implicit_clearance: {fmt(clearance.get('implicit_clearance'))}")
            print(f"    raw_implicit_clearance: {fmt(clearance.get('raw_implicit_clearance'))}")
            print(f"    fusion_total_risk_clearance: {fmt(fusion_clearance(clearance))}")
            print(f"    balanced_explicit_implicit_clearance: {fmt(balanced_clearance(clearance))}")


if __name__ == "__main__":
    print(f"Project root: {PROJ}")
    show_output_files()
    show_stage1()
    show_stage1_5()
    show_stage2()
    show_stage2_5()
    show_stage3()
