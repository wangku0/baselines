from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "integrations/my_method"))

from conversation import conv_templates
from infer_time_flow.controller import InferenceTimeFlowController
from utils import get_attn_implementation, resolve_dataset_path


random.seed(233)

SAFEERASER_TARGET_MODULES = (
    r".*language_model.*\."
    r"(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)"
)


def load_safeeraser_checkpoint(model, checkpoint_path: str, r: int = 32, alpha: int = 256):
    print(f"Loading SafeEraser checkpoint: {checkpoint_path}")
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=SAFEERASER_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected state-dict checkpoint at {checkpoint_path}, got {type(checkpoint).__name__}")
    model_keys = set(model.state_dict())
    matched_keys = model_keys.intersection(checkpoint)
    matched_lora_keys = [key for key in matched_keys if "lora_" in key]
    if not matched_lora_keys:
        raise RuntimeError(
            "The checkpoint did not match any SafeEraser LoRA parameters. "
            f"Sample checkpoint keys: {list(checkpoint)[:5]}"
        )
    incompatible = model.load_state_dict(checkpoint, strict=False)
    print(
        f"Loaded {len(matched_keys)}/{len(checkpoint)} checkpoint tensors "
        f"({len(matched_lora_keys)} LoRA tensors); "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    return model.merge_and_unload()


def _move_inputs(inputs, device, dtype):
    out = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            out[key] = value.to(device=device, dtype=dtype)
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _decode(processor, output, prompt_len: int) -> str:
    generated_ids = output[0, prompt_len:]
    decoded = processor.decode(generated_ids, skip_special_tokens=True).strip()
    if "ASSISTANT:" in decoded:
        decoded = decoded[decoded.index("ASSISTANT:") :].replace("ASSISTANT:", "", 1).strip()
    if decoded.endswith("</s>"):
        decoded = decoded[: -len("</s>")].strip()
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeEraser-format inference with inference-time Flow hidden-state intervention.")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--checkpoint_path", "--checkpoint-path", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    parser.add_argument("--flow_teacher_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=1,
        help="Number of sampled responses generated together for the same prompt/context.",
    )
    parser.add_argument("--device_map", choices=["auto", "single"], default="auto")
    parser.add_argument("--max_memory_per_gpu", default=None)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--risk_gate_threshold", type=float, default=0.0)
    parser.add_argument("--risk_gate_mode", choices=["fused", "implicit"], default="fused")
    parser.add_argument("--max_delta_norm_ratio", type=float, default=0.20)
    parser.add_argument(
        "--numerical_fallback_ratios",
        default="",
        help="Comma-separated max-delta ratios retried after a numerical generation error, e.g. 5,2.",
    )
    parser.add_argument("--risk_trace_max_records", type=int, default=200000)
    parser.add_argument("--no_prefill_intervention", action="store_true")
    parser.add_argument("--no_decode_intervention", action="store_true")
    args = parser.parse_args()
    if args.generation_batch_size < 1:
        raise ValueError("--generation_batch_size must be >= 1.")

    processor = AutoProcessor.from_pretrained(args.model_path)
    use_device_map = args.device_map == "auto"
    max_memory = None
    if use_device_map and args.max_memory_per_gpu:
        max_memory = {gpu_index: args.max_memory_per_gpu for gpu_index in range(torch.cuda.device_count())}
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation=get_attn_implementation(),
        torch_dtype=torch.float16,
        device_map="auto" if use_device_map else None,
        max_memory=max_memory,
    )
    if not use_device_map:
        model.half().to("cuda:0")
    if args.checkpoint_path:
        model = load_safeeraser_checkpoint(model, args.checkpoint_path)
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    controller = InferenceTimeFlowController(
        model,
        config_path=args.config,
        flow_teacher_path=args.flow_teacher_path,
        strength=args.strength,
        risk_gate_threshold=args.risk_gate_threshold,
        risk_gate_mode=args.risk_gate_mode,
        max_delta_norm_ratio=args.max_delta_norm_ratio,
        risk_trace_max_records=args.risk_trace_max_records,
        intervene_on_prefill=not args.no_prefill_intervention,
        intervene_on_decode=not args.no_decode_intervention,
    )
    controller.register()
    fallback_ratios = [
        float(value.strip())
        for value in args.numerical_fallback_ratios.split(",")
        if value.strip()
    ]
    if any(value < 0 for value in fallback_ratios):
        raise ValueError("--numerical_fallback_ratios values must be >= 0.")

    rows = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    results = []

    def generate_responses(
        prompt_text: str,
        image,
        *,
        num_responses: int,
        max_new_tokens: int,
        explicit_risk: float,
        group_id: int = 0,
    ) -> list[str]:
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = _move_inputs(inputs, input_device, model.dtype)
        preds = []
        while len(preds) < num_responses:
            current_batch = min(int(args.generation_batch_size), num_responses - len(preds))
            primary_ratio = controller.max_delta_norm_ratio
            retry_ratios = [primary_ratio] + [
                ratio for ratio in fallback_ratios if ratio != primary_ratio
            ]
            for attempt, ratio in enumerate(retry_ratios):
                controller.max_delta_norm_ratio = ratio
                try:
                    with controller.enabled(explicit_risk=explicit_risk, group_id=group_id):
                        output = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.9,
                            num_beams=1,
                            num_return_sequences=current_batch,
                        )
                    break
                except RuntimeError as exc:
                    message = str(exc).lower()
                    numerical_error = (
                        "probability tensor contains" in message
                        and ("inf" in message or "nan" in message or "element < 0" in message)
                    )
                    if not numerical_error or attempt + 1 >= len(retry_ratios):
                        raise
                    controller.stats.numerical_retries += 1
                    next_ratio = retry_ratios[attempt + 1]
                    print(
                        "Numerical generation error at "
                        f"max_delta_norm_ratio={ratio}; retrying current response with {next_ratio}."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                finally:
                    controller.max_delta_norm_ratio = primary_ratio
            prompt_len = inputs["input_ids"].shape[1]
            for row_idx in range(output.shape[0]):
                preds.append(_decode(processor, output[row_idx : row_idx + 1], prompt_len))
        return preds

    for line in tqdm(rows, desc="infer-time flow SafeEraser inference"):
        sd_image_path = resolve_dataset_path(line.get("SDImage_path"))
        sd_image = Image.open(sd_image_path).convert("RGB") if sd_image_path and os.path.exists(sd_image_path) else None
        image_id_path = resolve_dataset_path(line.get("image_path") or line.get("image_id"))
        id_image = Image.open(image_id_path).convert("RGB") if image_id_path and os.path.exists(image_id_path) else None

        output_line = dict(line)
        output_line["image_path"] = image_id_path
        if sd_image_path:
            output_line["SDImage_path"] = sd_image_path

        unsafe_pairs = output_line.get("unsafe_pairs", [])
        safe_nb_pairs = output_line.get("safeNb_pairs", [])

        if sd_image is not None:
            for pair in unsafe_pairs:
                q = pair.get("question", "")
                preds = generate_responses(
                    f"<image>\n{q}",
                    sd_image,
                    num_responses=1,
                    max_new_tokens=args.max_new_tokens,
                    explicit_risk=1.0,
                    group_id=0,
                )
                if preds:
                    pair["sd_response"] = preds[0]

        if id_image is not None:
            for key in ("UnharmPair_text1", "UnharmPair_text2", "UnharmPair_image1", "UnharmPair_image2"):
                item = output_line.get(key)
                if isinstance(item, dict) and str(item.get("Question", "")).strip():
                    preds = generate_responses(
                        f"<image>\n{item['Question']}",
                        id_image,
                        num_responses=1,
                        max_new_tokens=args.max_new_tokens,
                        explicit_risk=0.0,
                        group_id=2,
                    )
                    if preds:
                        item["Prediction"] = preds[0]

            for pair in unsafe_pairs:
                pair.pop("model_response", None)
                q = pair.get("question", "")
                preds = generate_responses(
                    f"<image>\n{q}",
                    id_image,
                    num_responses=3,
                    max_new_tokens=args.max_new_tokens,
                    explicit_risk=1.0,
                    group_id=0,
                )
                for idx, pred in enumerate(preds[:3], start=1):
                    pair[f"model_response{idx}"] = pred

            if isinstance(safe_nb_pairs, list):
                for pair in safe_nb_pairs:
                    if not isinstance(pair, dict):
                        continue
                    pair.pop("model_response", None)
                    q = str(pair.get("question", ""))
                    if q.strip():
                        preds = generate_responses(
                            f"<image>\n{q}",
                            id_image,
                            num_responses=1,
                            max_new_tokens=args.max_new_tokens,
                            explicit_risk=0.0,
                            group_id=1,
                        )
                        if preds:
                            pair["model_response1"] = preds[0]

        output_line["unsafe_pairs"] = unsafe_pairs
        if safe_nb_pairs:
            output_line["safeNb_pairs"] = safe_nb_pairs
        results.append(output_line)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path = out.with_suffix(".flow_stats.json")
    stats_path.write_text(json.dumps(controller.stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    trace_path = out.with_suffix(".flow_risk_trace.jsonl")
    controller.write_risk_trace(trace_path)
    print(f"Results saved to {out}")
    print(f"Flow intervention stats saved to {stats_path}")
    print(f"Flow risk trace saved to {trace_path}")


if __name__ == "__main__":
    main()
