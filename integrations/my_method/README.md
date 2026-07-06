# my_method integration for SafeEraser

This directory is isolated from the SafeEraser implementation. It adds the
`my_method` risk-space/Flow/LoRA pipeline while preserving the existing
SafeEraser training, inference and evaluation commands.

## Evaluation policy

- ASR, RR, SARR and ROUGE-L are produced only by the repository-root
  `ckpt_infer.py` and `eval_all.py`.
- Implicit risk is additional. It does not replace or alter any SafeEraser
  metric.
- Implicit risk is prompt-level: one score for each paired harmful/safeNb
  prompt. Fifty records produce 200 implicit-risk pairs, while SafeEraser
  produces 600 harmful-response classifications.

## Environment isolation

Use a separate environment for this directory. Do not install this
`requirements.txt` into the `safeeraser` environment.

```bash
python -m venv /kaggle/working/my_method_env
/kaggle/working/my_method_env/bin/python -m pip install -r integrations/my_method/requirements.txt
```

For GPU LLaMA-Guard, install the appropriate prebuilt `llama-cpp-python` wheel
instead of compiling the generic PyPI source package.

```bash
/kaggle/working/my_method_env/bin/python -m pip install \
  -r integrations/my_method/requirements-llama-guard.txt
```

## 1. Prepare exactly the same 50 records

Run from the repository root:

```bash
python integrations/my_method/scripts/prepare_safeeraser_data.py \
  --seed 233 \
  --records 50
```

This selects records from `dataset/all_train.json` using the same deterministic
half-sampling rule, then joins `dataset/paired/all_train.json` by `image_id`.
It writes only under `integrations/my_method/outputs/data/`.

Prepare the matching 50-record Violence validation subset from SafeEraser's
`dataset/all_val.json` and `dataset/paired/all_val.json`:

```bash
python integrations/my_method/scripts/prepare_safeeraser_val_data.py \
  --seed 233 \
  --records 50
```

This writes `violence_50_paired_val.json` for my_method hidden-state/risk
evaluation and `violence_50_val_eval.json` for unchanged SafeEraser inference.
Both files contain the same image IDs in the same order.

For a two-record end-to-end smoke test, use the checked-in inherited config:

```bash
python integrations/my_method/scripts/prepare_safeeraser_data.py --seed 233 --records 2
CFG=integrations/my_method/configs/safeeraser_llava_smoke.yaml
```

The smoke config inherits the production LLaVA settings, keeps
`stage2.response_source: generate`, uses the two-record paired train file for
both splits, reduces the risk-space sweep to k=1/2 and trains the Flow teacher
for 10 steps.

## 2. Build the LLaVA risk space

```bash
CFG=integrations/my_method/configs/safeeraser_llava.yaml
PY=/kaggle/working/my_method_env/bin/python

$PY integrations/my_method/scripts/01_extract_hidden_states.py --config $CFG --split train
$PY integrations/my_method/scripts/01_extract_hidden_states.py --config $CFG --split val
$PY integrations/my_method/scripts/02_build_risk_space.py --config $CFG
$PY integrations/my_method/scripts/03_evaluate_risk_space.py --config $CFG --split train
$PY integrations/my_method/scripts/03_evaluate_risk_space.py --config $CFG --split val
$PY integrations/my_method/scripts/04_stage1_5_analysis.py --config $CFG
$PY integrations/my_method/scripts/00_generate_stage2_base_responses.py --config $CFG --split both --force
$PY integrations/my_method/scripts/05_stage2_risk_evaluation.py --config $CFG --split both --skip_generation
$PY integrations/my_method/scripts/06_stage2_5_train_flow_matching.py --config $CFG --split train
```

The dedicated Stage 2 generation command creates fresh Base LLaVA responses
for harmful, safe-neighbor and retain prompts. Because those JSONL files then
exist, Stage 2 can either reuse them automatically or be run with
`--skip_generation` for a strict no-regeneration check before applying the
SafeEraser-aligned local LLaMA-Guard classifier. The existing Qwen risk
artifacts are not compatible with LLaVA; these commands create an isolated
LLaVA risk basis, selected layers and frozen normalization.

## 3. Train my_method

```bash
$PY integrations/my_method/scripts/06_stage3_train_lora_unlearning.py \
  --config $CFG
```

The adapter is stored under `integrations/my_method/outputs/`. No SafeEraser
checkpoint or configuration is overwritten.

## 4. Export and run unchanged SafeEraser evaluation

Merge the PEFT adapter into a complete FP16 LLaVA model:

```bash
$PY integrations/my_method/scripts/export_merged_llava.py \
  --adapter-path integrations/my_method/outputs/stage3/lora_flow_unlearned/adapter \
  --output-dir integrations/my_method/outputs/export/merged_llava
```

Then switch to the existing SafeEraser environment and invoke the unchanged
root commands through the bridge:

```bash
/root/miniconda3/envs/safeeraser/bin/python \
  integrations/my_method/scripts/run_safeeraser_evaluation.py \
  --model-path integrations/my_method/outputs/export/merged_llava \
  --method-name my_method \
  --python /root/miniconda3/envs/safeeraser/bin/python
```

## 5. Add implicit risk

```bash
$PY integrations/my_method/scripts/score_implicit_risk.py \
  --config $CFG \
  --adapter-path integrations/my_method/outputs/stage3/lora_flow_unlearned/adapter \
  --method-name my_method \
  --split train
```

The same scorer can later evaluate a SafeEraser method without modifying its
code:

```bash
$PY integrations/my_method/scripts/score_implicit_risk.py \
  --config $CFG \
  --safeeraser-checkpoint ckpt_results/llava-7b/METHOD/checkpoint.pt \
  --method-name METHOD \
  --split train
```

## 6. Combine reports

```bash
$PY integrations/my_method/scripts/combine_reports.py \
  --safeeraser-summary integrations/my_method/outputs/unified_eval/my_method_safeeraser_evaluated.summary.json \
  --implicit-summary integrations/my_method/outputs/unified_eval/my_method_implicit_summary.json \
  --method-name my_method \
  --output integrations/my_method/outputs/unified_eval/my_method_final_report.json
```

## Zero-intrusion check

All runtime products must remain under `integrations/my_method/outputs/`.
Existing SafeEraser `.py`, `.yaml` and `.bash` files are not changed by this
integration.
