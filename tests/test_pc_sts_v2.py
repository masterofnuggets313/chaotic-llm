import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phase01", "exp_vq"))
from models_pc_v2 import PCSTSLMv2


def test_pc_sts_v2_forward_and_selection():
    model = PCSTSLMv2(vocab=97, d=32, layers=2, window=32, topk=4)
    logits = model(torch.randint(0, 97, (3, 32)))
    assert logits.shape == (3, 97)
    assert torch.isfinite(logits).all()
    assert model._last_selection["indices"].shape == (3, 4)
    assert (model._last_selection["indices"] < 24).all()


def test_pc_sts_v2_starts_with_raw_retrieval_score_only():
    model = PCSTSLMv2(vocab=97, d=32, layers=1, window=32)
    model(torch.randint(0, 97, (1, 32)))
    selected = model._last_selection
    indices = selected["indices"]
    assert torch.allclose(selected["score"].gather(1, indices),
                          selected["raw_score"].gather(1, indices), atol=1e-6)
