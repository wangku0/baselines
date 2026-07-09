import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.flow_matching.features import _flow_target_config, build_flow_features
from src.flow_matching.train import train_flow_teacher
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, ensure_dir, load_config, resolve_path, save_json


def _materialize_flow_metadata(config, *, split, max_pairs, debug, recommended_config_path):
    """Use existing flow_teacher.pt but regenerate lightweight metadata files."""
    flow_out = ensure_dir(resolve_path(config, config.get("flow_matching", {}).get("output_dir", "integrations/my_method/outputs/stage2_5_flow")))
    feature_path = build_flow_features(
        config,
        split=split,
        max_pairs=max_pairs,
        debug=debug,
        recommended_config_path=recommended_config_path,
    )
    feature_data = torch.load(feature_path, map_location="cpu", weights_only=False)
    teacher_path = flow_out / "flow_teacher.pt"
    teacher_data = torch.load(teacher_path, map_location="cpu", weights_only=False)
    train_log = flow_out / "train_log.csv"
    eval_summary = {
        "reused_flow_teacher": True,
        "flow_teacher_path": str(teacher_path),
        "features_path": str(feature_path),
        "recommended": feature_data.get("recommended", teacher_data.get("recommended")),
        "flow_target": feature_data.get("flow_target", teacher_data.get("flow_target")),
        "train_log": str(train_log) if train_log.exists() else None,
    }
    save_json(eval_summary, flow_out / "eval_summary.json")
    with (flow_out / "flow_config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "flow_matching": config.get("flow_matching", {}),
                "recommended": eval_summary["recommended"],
                "hidden_dim": int(feature_data.get("hidden_dim", teacher_data.get("hidden_dim", 0))),
                "cond_dim": int(feature_data.get("cond_dim", teacher_data.get("cond_dim", 0))),
                "flow_target": eval_summary["flow_target"],
                "representation_pooling": feature_data.get("representation_pooling"),
                "requested_representation_pooling": feature_data.get("requested_representation_pooling"),
                "reused_flow_teacher": True,
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )
    print("Flow teacher exists; regenerated Stage 2.5 metadata/evaluation files from cached artifacts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2.5 hidden-space flow matching teacher.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--recommended_config_path", default=None)
    parser.add_argument(
        "--flow_output_dir",
        default=None,
        help="Override flow_matching.output_dir so target ablations can be saved in separate folders.",
    )
    parser.add_argument(
        "--flow_target",
        choices=["safe_neighbor", "safenb", "safe", "retain", "mixed", "mix"],
        default=None,
        help="Override flow_matching.target.mode. Default keeps YAML behavior.",
    )
    parser.add_argument(
        "--flow_target_safe_weight",
        type=float,
        default=None,
        help="SafeNb weight for --flow_target mixed. The script normalizes safe/retain weights.",
    )
    parser.add_argument(
        "--flow_target_retain_weight",
        type=float,
        default=None,
        help="Retain weight for --flow_target mixed. The script normalizes safe/retain weights.",
    )
    parser.add_argument("--reuse_existing", action="store_true", help="Skip if flow_teacher.pt already exists.")
    parser.add_argument("--force", action="store_true", help="Force retraining even when --reuse_existing products exist.")
    parser.add_argument(
        "--layer_selection_method",
        choices=["stage1_5_ablation", "stage1_5_recommended", "risk_transport_influence"],
        default=None,
        help="Override stage3.layer_selection.method for Flow teacher feature construction.",
    )
    parser.add_argument(
        "--layer_selection_top_n",
        default=None,
        help="Override stage3.layer_selection.top_n for risk_transport_influence. Use an integer or 'auto'.",
    )
    parser.add_argument(
        "--transport_target",
        choices=["safe_neighbor", "safenb", "safe", "retain", "mixed", "mix"],
        default=None,
        help="Override stage3.layer_selection.transport_target.mode for risk_transport_influence.",
    )
    parser.add_argument("--transport_target_safe_weight", type=float, default=None)
    parser.add_argument("--transport_target_retain_weight", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if args.flow_output_dir:
        config.setdefault("flow_matching", {})["output_dir"] = args.flow_output_dir
    if args.flow_target:
        config.setdefault("flow_matching", {}).setdefault("target", {})["mode"] = args.flow_target
    if args.flow_target_safe_weight is not None:
        config.setdefault("flow_matching", {}).setdefault("target", {})["safe_weight"] = float(args.flow_target_safe_weight)
    if args.flow_target_retain_weight is not None:
        config.setdefault("flow_matching", {}).setdefault("target", {})["retain_weight"] = float(args.flow_target_retain_weight)
    if args.layer_selection_method:
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["method"] = args.layer_selection_method
    if args.layer_selection_top_n is not None:
        raw_top_n = str(args.layer_selection_top_n)
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["top_n"] = (
            raw_top_n if raw_top_n.lower() == "auto" else int(raw_top_n)
        )
    if args.transport_target is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})["mode"] = args.transport_target
    if args.transport_target_safe_weight is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})["safe_weight"] = float(args.transport_target_safe_weight)
    if args.transport_target_retain_weight is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})["retain_weight"] = float(args.transport_target_retain_weight)
    flow_out = config.get("flow_matching", {}).get("output_dir", "integrations/my_method/outputs/stage2_5_flow")
    target_cfg = _flow_target_config(config)
    teacher_path = resolve_path(config, f"{flow_out}/flow_teacher.pt")
    force_for_target_mismatch = False
    if args.reuse_existing and not args.force and teacher_path.exists():
        try:
            existing = torch.load(teacher_path, map_location="cpu", weights_only=False)
            existing_dynamic = existing.get("dynamic_conditioning") or {}
            if not bool(existing_dynamic.get("R_imp_norm_t", False)):
                print("[reuse_existing] Stage 2.5 Flow teacher: existing teacher lacks dynamic R_imp_norm(t); recomputing.")
                force_for_target_mismatch = True
            elif existing_dynamic.get("normalization") != "stage2_sample_risk":
                print(
                    "[reuse_existing] Stage 2.5 Flow teacher: existing teacher uses the old "
                    "dynamic-risk normalization; recomputing."
                )
                force_for_target_mismatch = True
            existing_target = existing.get("flow_target")
            if args.flow_target is not None and not existing_target:
                print("[reuse_existing] Stage 2.5 Flow teacher: existing teacher has no flow_target metadata; recomputing.")
                print(f"  - requested: {target_cfg}")
                force_for_target_mismatch = True
            elif args.flow_target is not None and (
                existing_target.get("mode") != target_cfg.get("mode")
                or abs(float(existing_target.get("safe_weight", 0.0)) - float(target_cfg.get("safe_weight", 0.0))) > 1e-6
                or abs(float(existing_target.get("retain_weight", 0.0)) - float(target_cfg.get("retain_weight", 0.0))) > 1e-6
            ):
                print("[reuse_existing] Stage 2.5 Flow teacher: existing target differs from requested target; recomputing.")
                print(f"  - existing: {existing_target}")
                print(f"  - requested: {target_cfg}")
                force_for_target_mismatch = True
        except Exception as exc:
            print(f"[reuse_existing] Could not inspect existing flow teacher target ({exc}); recomputing.")
            force_for_target_mismatch = True
    if reuse_if_exists(
        config,
        [f"{flow_out}/flow_teacher.pt"],
        label="Stage 2.5 Flow teacher",
        reuse_existing=args.reuse_existing,
        force=args.force or force_for_target_mismatch,
    ):
        _materialize_flow_metadata(
            config,
            split=args.split,
            max_pairs=args.max_pairs,
            debug=args.debug,
            recommended_config_path=args.recommended_config_path,
        )
        print(f"Flow teacher reused from {Path(flow_out) / 'flow_teacher.pt'}")
        return
    path = train_flow_teacher(
        config,
        split=args.split,
        max_pairs=args.max_pairs,
        batch_size=args.batch_size,
        steps=args.steps,
        device=args.device,
        debug=args.debug,
        recommended_config_path=args.recommended_config_path,
    )
    print(f"Flow teacher saved to {path}")


if __name__ == "__main__":
    main()
