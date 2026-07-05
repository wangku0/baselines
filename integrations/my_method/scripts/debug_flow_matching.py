import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.flow_matching.model import FlowVectorField, euler_integrate_flow
from src.flow_matching.utils import lambda_flow_ramp, response_mean_last_pool
from src.risk_space.recommended_config import hidden_layers_to_lora_layers, load_recommended_risk_config
from src.utils import load_config


def main() -> None:
    cfg = load_config("integrations/my_method/configs/safeeraser_llava.yaml")
    try:
        rec = load_recommended_risk_config(cfg, allow_fallback=True)
        print("recommended:", rec.to_dict())
    except Exception as exc:
        print("recommended loader check skipped:", exc)
    assert hidden_layers_to_lora_layers([16]) == [15]
    assert hidden_layers_to_lora_layers([1]) == [0]
    try:
        hidden_layers_to_lora_layers([0])
        raise AssertionError("hidden layer 0 should fail")
    except ValueError:
        pass
    h = torch.randn(2, 5, 7)
    mask = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
    pooled = response_mean_last_pool(h, mask)
    assert pooled.shape == (2, 7)
    model = FlowVectorField(hidden_dim=7, cond_dim=4, hidden_width=16, time_embedding_dim=8)
    x = torch.randn(3, 7)
    c = torch.randn(3, 4)
    t = torch.rand(3, 1)
    y = model(x, t, c)
    assert y.shape == x.shape
    z = euler_integrate_flow(model, x, c, steps=2)
    assert z.shape == x.shape
    assert lambda_flow_ramp(0, 100, 0.5, 0.1, 0.4) == 0.0
    assert abs(lambda_flow_ramp(50, 100, 0.5, 0.1, 0.4) - 0.5) < 1e-6
    print("Flow matching sanity checks passed.")


if __name__ == "__main__":
    main()

