# Inference-Time Flow Intervention

This folder is isolated from the main `my_method` training path. It reuses the
existing risk-space, Flow teacher, SafeEraser inference format, and evaluation
utilities, but writes outputs under:

```text
integrations/my_method/infer_time_flow/outputs/
```

The controller loads the base model and applies forward hooks on the selected
risk layers during generation. At each hooked layer it computes the current
Stage2-normalized `R_imp_norm(t)`, builds an inference-time Flow condition, and
adds a gated Flow velocity to the last-token hidden state.

The intervention gate uses the same fused-risk spirit as the main method:

```text
R_total(t) = 0.5 * R_exp + 0.5 * R_imp_norm(t)
gate = clip((R_total(t) - risk_gate_threshold) / (1 - risk_gate_threshold), 0, 1)
h_l' = h_l + strength * gate * velocity
```

For SafeEraser-format generation, harmful/SD prompts use `R_exp=1`, while
safe-neighbor and retain prompts use `R_exp=0`.

Run:

```bash
python integrations/my_method/infer_time_flow/run_evaluation.py \
  --model-path llava-hf/llava-1.5-7b-hf \
  --config integrations/my_method/configs/safeeraser_llava.yaml \
  --flow-teacher-path integrations/my_method/outputs/stage2_5_flow/flow_teacher.pt \
  --eval-file integrations/my_method/outputs/data/violence_50_val_eval.json \
  --method-name infer_time_flow_val50 \
  --expected-records 50 \
  --max-new-tokens 256
```

Outputs:

```text
infer_time_flow_val50_predictions.json
infer_time_flow_val50_safeeraser_evaluated.json
infer_time_flow_val50_safeeraser_evaluated.summary.json
infer_time_flow_val50_implicit_scores.csv
infer_time_flow_val50_implicit_summary.json
infer_time_flow_val50_final_report.json
```

Note: the trained Flow teacher used paired safe-neighbor targets during
training. At inference time the paired target is not available, so this
controller approximates the transport condition from the current hidden state's
risk-subspace coordinates and dynamic `R_imp_norm(t)`.
