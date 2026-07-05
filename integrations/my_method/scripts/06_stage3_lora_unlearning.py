import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage3_trainer import train_stage3_lora
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 risk-subspace guided LoRA unlearning.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--safe_neighbor_ce_weight", type=float, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume_from_adapter", default=None)
    parser.add_argument("--reuse_existing", action="store_true", help="Skip training if adapter_config.json already exists.")
    parser.add_argument("--force", action="store_true", help="Force retraining even when --reuse_existing products exist.")
    parser.add_argument(
        "--layer_selection_method",
        choices=["stage1_5_ablation", "stage1_5_recommended", "risk_transport_influence"],
        default=None,
        help="Override stage3.layer_selection.method without editing the YAML config.",
    )
    parser.add_argument(
        "--layer_selection_top_n",
        default=None,
        help="Override stage3.layer_selection.top_n for risk_transport_influence. Use an integer or 'auto'.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if args.safe_neighbor_ce_weight is not None:
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["safe_neighbor_ce"] = float(args.safe_neighbor_ce_weight)
    if args.layer_selection_method:
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["method"] = args.layer_selection_method
    if args.layer_selection_top_n is not None:
        raw_top_n = str(args.layer_selection_top_n)
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["top_n"] = (
            raw_top_n if raw_top_n.lower() == "auto" else int(raw_top_n)
        )
    adapter_dir = Path(args.output_dir) / "adapter" if args.output_dir else Path(config["stage3"]["outputs"]["adapter_dir"])
    if reuse_if_exists(
        config,
        [adapter_dir / "adapter_config.json"],
        label="Stage 3 LoRA adapter",
        reuse_existing=args.reuse_existing,
        force=args.force,
    ):
        return
    result = train_stage3_lora(
        config,
        max_steps=args.max_steps,
        max_train_samples=args.max_train_samples,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        debug=args.debug,
        resume_from_adapter=args.resume_from_adapter,
    )
    if result.get("trained_steps", 0) == 0:
        print("Stage 3 data construction finished.")
        print(f"Triplets: {result.get('triplets')}")


if __name__ == "__main__":
    main()
