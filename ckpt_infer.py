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

    def generate_responses(prompt_text, image,
                           num_responses=1,
                           temperature=1.0,
                           top_p=0.9,
                           num_beams=1,
                          my_max_new_tokens=512):

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            key: (
                value.to(device=input_device, dtype=model.dtype)
                if torch.is_floating_point(value)
                else value.to(input_device)
            )
            for key, value in inputs.items()
        }

        all_predictions = []
        for _ in range(num_responses):
            output = model.generate(
                **inputs,
                max_new_tokens=my_max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams
            )
            generated_ids = output[0, inputs["input_ids"].shape[1]:]
            decoded = processor.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            if "ASSISTANT:" in decoded:
                idx = decoded.index("ASSISTANT:")
                prediction = decoded[idx:].replace("ASSISTANT:", "", 1).strip()
            else:
                prediction = decoded


            if prediction.endswith("</s>"):
                prediction = prediction[:-len("</s>")].strip()

            all_predictions.append(prediction)

        return all_predictions

    for line in tqdm(data):
        sd_image_path = resolve_dataset_path(line.get("SDImage_path"))
        if sd_image_path and os.path.exists(sd_image_path):
            sd_image = Image.open(sd_image_path).convert('RGB')
            sd_image_input = sd_image
        else:
            print("sd_image_input is None")
            sd_image_input = None


        image_id_path = resolve_dataset_path(
            line.get("image_path") or line.get("image_id")
        )
        if image_id_path and os.path.exists(image_id_path):
            id_image = Image.open(image_id_path).convert('RGB')
            id_image_input = id_image
        else:
            print("id_image_input is None")
            id_image_input = None

        output_line = {}
        for k, v in line.items():
            output_line[k] = v
        output_line["image_path"] = image_id_path
        if sd_image_path:
            output_line["SDImage_path"] = sd_image_path

        unsafe_pairs = output_line.get("unsafe_pairs", [])
        if sd_image_input is not None:
            for up in unsafe_pairs:
                q = up.get("question", "")

                prompt_text = f"<image>\n{q}"
                new_sd_resp = generate_responses(prompt_text, sd_image_input, num_responses=1,my_max_new_tokens=tokens_sd)

                if new_sd_resp:
                    up["sd_response"] = new_sd_resp[0]

        if id_image_input is not None:
            if "UnharmPair_text1" in output_line and isinstance(output_line["UnharmPair_text1"], dict):
                q_text1 = output_line["UnharmPair_text1"].get("Question", "")
                if q_text1.strip():
                    prompt_text1 = f"<image>\n{q_text1}"
                    new_pred_text1 = generate_responses(prompt_text1, id_image_input, num_responses=1,my_max_new_tokens=tokens_text)
                    if new_pred_text1:
                        output_line["UnharmPair_text1"]["Prediction"] = new_pred_text1[0]

            if "UnharmPair_text2" in output_line and isinstance(output_line["UnharmPair_text2"], dict):
                q_text2 = output_line["UnharmPair_text2"].get("Question", "")
                if q_text2.strip():
                    prompt_text2 = f"<image>\n{q_text2}"
                    new_pred_text2 = generate_responses(prompt_text2, id_image_input, num_responses=1,my_max_new_tokens=tokens_text)
                    if new_pred_text2:
                        output_line["UnharmPair_text2"]["Prediction"] = new_pred_text2[0]

            if "UnharmPair_image1" in output_line and isinstance(output_line["UnharmPair_image1"], dict):
                q_img1 = output_line["UnharmPair_image1"].get("Question", "")
                if q_img1.strip():
                    prompt_img1 = f"<image>\n{q_img1}"
                    new_pred_img1 = generate_responses(prompt_img1, id_image_input, num_responses=1,my_max_new_tokens=tokens_image)
                    if new_pred_img1:
                        output_line["UnharmPair_image1"]["Prediction"] = new_pred_img1[0]

            if "UnharmPair_image2" in output_line and isinstance(output_line["UnharmPair_image2"], dict):
                q_img2 = output_line["UnharmPair_image2"].get("Question", "")
                if q_img2.strip():
                    prompt_img2 = f"<image>\n{q_img2}"
                    new_pred_img2 = generate_responses(prompt_img2, id_image_input, num_responses=1,my_max_new_tokens=tokens_image)
                    if new_pred_img2:
                        output_line["UnharmPair_image2"]["Prediction"] = new_pred_img2[0]

        if id_image_input is not None:
            for up in unsafe_pairs:
                if "model_response" in up:
                    del up["model_response"]

                q = up.get("question", "")
                prompt_text = f"<image>\n{q}"
                model_preds = generate_responses(prompt_text, id_image_input, num_responses=3,my_max_new_tokens=tokens_harm)
                if len(model_preds) >= 1:
                    up["model_response1"] = model_preds[0]
                if len(model_preds) >= 2:
                    up["model_response2"] = model_preds[1]
                if len(model_preds) >= 3:
                    up["model_response3"] = model_preds[2]

        output_line["unsafe_pairs"] = unsafe_pairs

        results.append(output_line)

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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="maximum generated tokens per response (default: 256)",
    )
    args = parser.parse_args()
    main(args)
