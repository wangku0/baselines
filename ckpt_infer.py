import os
import json
import torch
import random
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse
from peft import LoraConfig, get_peft_model
from conversation import conv_templates  
from utils import get_attn_implementation, resolve_dataset_path

random.seed(233)

SAFE_EMPTY_RESPONSE = "I cannot provide a response."


def main(args):
    file = args.eval_file
    model_path = args.model_path
    output_file = args.output_file
    tokens_text = args.max_new_tokens
    tokens_image = args.max_new_tokens
    tokens_sd = args.max_new_tokens
    tokens_harm = args.max_new_tokens


    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    use_device_map = args.device_map == "auto"
    max_memory = None
    if use_device_map and args.max_memory_per_gpu:
        max_memory = {
            gpu_index: args.max_memory_per_gpu
            for gpu_index in range(torch.cuda.device_count())
        }

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation=get_attn_implementation(),
        torch_dtype=torch.float16,
        device_map="auto" if use_device_map else None,
        max_memory=max_memory,
    )
    if args.checkpoint_path is not None:
        print("Merging Lora Weights.....")
        target_modules = r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'
        config = LoraConfig(
            r=32,
            lora_alpha=256,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Expected a state-dict checkpoint at {args.checkpoint_path}, "
                f"got {type(checkpoint).__name__}"
            )
        model_keys = set(model.state_dict())
        matched_keys = model_keys.intersection(checkpoint)
        matched_lora_keys = [key for key in matched_keys if "lora_" in key]
        if not matched_lora_keys:
            sample_keys = list(checkpoint)[:5]
            raise RuntimeError(
                "The checkpoint did not match any LoRA parameters in the "
                f"inference model. Sample checkpoint keys: {sample_keys}"
            )
        incompatible = model.load_state_dict(checkpoint, strict=False)
        print(
            f"Loaded {len(matched_keys)}/{len(checkpoint)} checkpoint tensors "
            f"({len(matched_lora_keys)} LoRA tensors); "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        model = model.merge_and_unload()

    if not use_device_map:
        model.half().to("cuda:0")
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    if use_device_map:
        print(f"Model device map: {model.hf_device_map}")
    else:
        print("Model device: cuda:0")

    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    def _build_prompt(prompt_text):
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _move_inputs(inputs):
        return {
            key: (
                value.to(device=input_device, dtype=model.dtype)
                if torch.is_floating_point(value)
                else value.to(input_device)
            )
            for key, value in inputs.items()
        }

    def _decode_prediction(generated_ids):
        decoded = processor.decode(generated_ids, skip_special_tokens=True).strip()
        if "ASSISTANT:" in decoded:
            idx = decoded.index("ASSISTANT:")
            prediction = decoded[idx:].replace("ASSISTANT:", "", 1).strip()
        else:
            prediction = decoded
        if prediction.endswith("</s>"):
            prediction = prediction[:-len("</s>")].strip()
        return prediction if prediction.strip() else SAFE_EMPTY_RESPONSE

    generation_tasks = []

    def add_generation_task(prompt_text, image_path, target, key, max_new_tokens):
        if not image_path or not os.path.exists(image_path):
            return
        generation_tasks.append(
            {
                "prompt": _build_prompt(prompt_text),
                "image_path": image_path,
                "target": target,
                "key": key,
                "max_new_tokens": int(max_new_tokens),
            }
        )

    for line in tqdm(data, desc="prepare ckpt eval tasks"):
        sd_image_path = resolve_dataset_path(line.get("SDImage_path"))
        if not (sd_image_path and os.path.exists(sd_image_path)):
            print("sd_image_input is None")
            sd_image_path = None

        image_id_path = resolve_dataset_path(line.get("image_path") or line.get("image_id"))
        if not (image_id_path and os.path.exists(image_id_path)):
            print("id_image_input is None")
            image_id_path = None

        output_line = {k: v for k, v in line.items()}
        output_line["image_path"] = image_id_path
        if sd_image_path:
            output_line["SDImage_path"] = sd_image_path

        unsafe_pairs = output_line.get("unsafe_pairs", [])
        safe_nb_pairs = output_line.get("safeNb_pairs", [])

        if sd_image_path is not None:
            for up in unsafe_pairs:
                q = up.get("question", "")
                add_generation_task(f"<image>\n{q}", sd_image_path, up, "sd_response", tokens_sd)

        if image_id_path is not None:
            retain_fields = (
                ("UnharmPair_text1", tokens_text),
                ("UnharmPair_text2", tokens_text),
                ("UnharmPair_image1", tokens_image),
                ("UnharmPair_image2", tokens_image),
            )
            for field, max_tokens in retain_fields:
                value = output_line.get(field)
                if isinstance(value, dict):
                    q = str(value.get("Question", "")).strip()
                    if q:
                        add_generation_task(f"<image>\n{q}", image_id_path, value, "Prediction", max_tokens)

            for up in unsafe_pairs:
                if "model_response" in up:
                    del up["model_response"]
                q = up.get("question", "")
                prompt_text = f"<image>\n{q}"
                add_generation_task(prompt_text, image_id_path, up, "model_response1", tokens_harm)
                add_generation_task(prompt_text, image_id_path, up, "model_response2", tokens_harm)
                add_generation_task(prompt_text, image_id_path, up, "model_response3", tokens_harm)

            if isinstance(safe_nb_pairs, list):
                for sp in safe_nb_pairs:
                    if not isinstance(sp, dict):
                        continue
                    if "model_response" in sp:
                        del sp["model_response"]
                    q = str(sp.get("question", "")).strip()
                    if q:
                        add_generation_task(f"<image>\n{q}", image_id_path, sp, "model_response1", tokens_harm)

        output_line["unsafe_pairs"] = unsafe_pairs
        if safe_nb_pairs:
            output_line["safeNb_pairs"] = safe_nb_pairs
        results.append(output_line)

    def run_generation_batch(task_batch):
        images = [Image.open(task["image_path"]).convert("RGB") for task in task_batch]
        prompts = [task["prompt"] for task in task_batch]
        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt")
        inputs = _move_inputs(inputs)
        max_new_tokens = int(task_batch[0]["max_new_tokens"])
        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    num_beams=1,
                    num_return_sequences=1,
                )
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass
        prompt_len = int(inputs["input_ids"].shape[1])
        for row_idx, task in enumerate(task_batch):
            generated_ids = output[row_idx, prompt_len:]
            task["target"][task["key"]] = _decode_prediction(generated_ids)

    def run_generation_batch_with_fallback(task_batch):
        if not task_batch:
            return
        try:
            run_generation_batch(task_batch)
        except RuntimeError as exc:
            message = str(exc).lower()
            if len(task_batch) > 1 and ("out of memory" in message or "cuda" in message):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mid = len(task_batch) // 2
                run_generation_batch_with_fallback(task_batch[:mid])
                run_generation_batch_with_fallback(task_batch[mid:])
                return
            raise

    batch_size = max(1, int(args.generation_batch_size))
    for max_new_tokens in sorted({task["max_new_tokens"] for task in generation_tasks}):
        token_tasks = [task for task in generation_tasks if task["max_new_tokens"] == max_new_tokens]
        for start in tqdm(
            range(0, len(token_tasks), batch_size),
            desc=f"ckpt infer cross-sample generate max_tokens={max_new_tokens}",
        ):
            run_generation_batch_with_fallback(token_tasks[start : start + batch_size])

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True, help="the path to the eval file")
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf", help="model path")
    parser.add_argument("--output_file", type=str, required=True, help="path to save the output results")
    parser.add_argument("--checkpoint_path", choices=None, default=None, type=str, help="lora weights of unlearning methods")
    parser.add_argument("--loss_type", choices=["ga", "kl", "po",  "retain", "full","gd","gapd","gdpd","klpd","popd","idk","idkpd"], default="ga", type=str, help="unlearning method")
    parser.add_argument(
        "--device_map",
        choices=["auto", "single"],
        default="auto",
        help="auto shards the model across all visible GPUs; single uses cuda:0",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        default=None,
        help="optional per-GPU limit used by device_map=auto, e.g. 13GiB",
    )
    parser.add_argument("--gpu_memory", default=None, help="Alias for --max_memory_per_gpu, e.g. 75GiB")
    parser.add_argument("--a800_75g", action="store_true", help="Convenience preset: --max_memory_per_gpu 75GiB")
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=1,
        help="Number of sampled responses generated together for the same prompt/context.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Alias for --generation_batch_size")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="maximum generated tokens per response (default: 256)",
    )
    args = parser.parse_args()
    if args.batch_size is not None:
        args.generation_batch_size = int(args.batch_size)
    if args.generation_batch_size < 1:
        raise ValueError("--generation_batch_size must be >= 1.")
    if args.a800_75g:
        args.max_memory_per_gpu = "75GiB"
    if args.gpu_memory is not None:
        args.max_memory_per_gpu = str(args.gpu_memory)
    main(args)
