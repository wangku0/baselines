import argparse
import sys
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_space.transport_layer_selection import run_risk_transport_topn_ablation
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate and recommend risk-transport layer-selection top_n.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--recommended_config_path", default=None)
    parser.add_argument("--max_top_n", type=int, default=None)
    parser.add_argument("--coverage_target", type=float, default=None)
    parser.add_argument("--complexity_penalty", type=float, default=None)
    parser.add_argument("--retain_penalty", type=float, default=None)
    parser.add_argument("--min_marginal_gain", type=float, default=None)
    parser.add_argument(
        "--transport_target",
        choices=["safe_neighbor", "safenb", "safe", "retain", "mixed", "mix"],
        default=None,
        help="Override stage3.layer_selection.transport_target.mode.",
    )
    parser.add_argument("--transport_target_safe_weight", type=float, default=None)
    parser.add_argument("--transport_target_retain_weight", type=float, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--reuse_existing", action="store_true", help="Reuse the existing top_n ablation JSON if it exists.")
    parser.add_argument("--force", action="store_true", help="Force recomputation even when --reuse_existing output exists.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    sel_cfg = config.setdefault("stage3", {}).setdefault("layer_selection", {})
    sel_cfg["method"] = "risk_transport_influence"
    if args.coverage_target is not None:
        sel_cfg["top_n_coverage_target"] = float(args.coverage_target)
    if args.complexity_penalty is not None:
        sel_cfg["top_n_complexity_penalty"] = float(args.complexity_penalty)
    if args.retain_penalty is not None:
        sel_cfg["top_n_retain_penalty"] = float(args.retain_penalty)
    if args.min_marginal_gain is not None:
        sel_cfg["top_n_min_marginal_gain"] = float(args.min_marginal_gain)
    if args.transport_target is not None:
        sel_cfg.setdefault("transport_target", {})["mode"] = args.transport_target
    if args.transport_target_safe_weight is not None:
        sel_cfg.setdefault("transport_target", {})["safe_weight"] = float(args.transport_target_safe_weight)
    if args.transport_target_retain_weight is not None:
        sel_cfg.setdefault("transport_target", {})["retain_weight"] = float(args.transport_target_retain_weight)
    if args.output_path:
        sel_cfg["top_n_ablation_path"] = args.output_path
    output_path = sel_cfg.get("top_n_ablation_path") or "integrations/my_method/outputs/metrics/stage3/risk_transport_topn_ablation.json"

    if reuse_if_exists(
        config,
        [output_path],
        label="Risk-transport top_n ablation",
        reuse_existing=args.reuse_existing,
        force=args.force,
    ):
        path = Path(output_path)
        if not path.is_absolute():
            from src.utils import resolve_path

            path = resolve_path(config, output_path)
        result = json.load(path.open("r", encoding="utf-8"))
        print("Risk-transport top_n ablation reused.")
        print(f"recommended_top_n: {result['recommended_top_n']}")
        print(f"recommended_hidden_layers: {result['recommended_hidden_layers']}")
        print(f"recommended_lora_layers: {result['recommended_lora_layers']}")
        print(f"reason: {result['reason']}")
        return

    result = run_risk_transport_topn_ablation(
        config,
        recommended_config_path=args.recommended_config_path,
        max_top_n=args.max_top_n,
    )
    print("Risk-transport top_n ablation finished.")
    print(f"recommended_top_n: {result['recommended_top_n']}")
    print(f"recommended_hidden_layers: {result['recommended_hidden_layers']}")
    print(f"recommended_lora_layers: {result['recommended_lora_layers']}")
    print(f"reason: {result['reason']}")


if __name__ == "__main__":
    main()
