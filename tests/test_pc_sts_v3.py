import os
import sys
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phase01", "exp_vq"))

from models_pc_v3 import (
    RMSNorm,
    SwiGLUFFN,
    MultiHeadSparseRetrieval,
    VectorGatedPCBlock,
    PCSTSLMv3,
    count_params
)


def test_rmsnorm():
    norm = RMSNorm(32)
    x = torch.randn(4, 16, 32) * 5.0
    out = norm(x)
    assert out.shape == (4, 16, 32)
    assert torch.isfinite(out).all()
    # RMS по последней оси должно быть примерно 1.0 (умножено на weight=1.0)
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_swiglu_ffn():
    ffn = SwiGLUFFN(32, multiplier=2.0)
    x = torch.randn(2, 16, 32)
    out = ffn(x)
    assert out.shape == (2, 16, 32)
    assert torch.isfinite(out).all()


def test_multihead_sparse_retrieval():
    retrieval = MultiHeadSparseRetrieval(d=32, num_heads=4, topk=4, temp=0.3)
    raw_e = torch.randn(2, 32, 32)
    q_current = torch.randn(2, 32)
    driver, top_idx = retrieval(raw_e, q_current, length=32)

    assert driver.shape == (2, 1, 32)
    assert top_idx.shape == (2, 4, 4)  # [B, H, K]
    assert torch.isfinite(driver).all()
    assert (top_idx < 24).all()  # т.к. хвост 8 маскируется (32 - 8 = 24)


def test_vector_gated_pc_block():
    block = VectorGatedPCBlock(d=32, num_heads=4, topk=4, alpha=0.3)
    h = torch.randn(2, 32, 32)
    raw_e = h.clone()
    q_current = h[:, -1]
    out, top_idx = block(h, raw_e, q_current, length=32)

    assert out.shape == (2, 32, 32)
    assert torch.isfinite(out).all()
    assert top_idx.shape == (2, 4, 4)


def test_pc_sts_v3_full_model():
    model = PCSTSLMv3(vocab=128, d=32, layers=2, window=64, num_heads=4, topk=4)
    x = torch.randint(0, 128, (3, 64))
    logits = model(x)

    assert logits.shape == (3, 128)
    assert torch.isfinite(logits).all()
    assert model._last_selection["indices"].shape == (3, 4, 4)

    # Проверка обратного прохода (Gradient Flow)
    loss = logits.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Градиент не поступил в {name}"
            assert torch.isfinite(param.grad).all(), f"NaN/Inf градиент в {name}"


def test_pc_sts_v3_variable_length():
    model = PCSTSLMv3(vocab=64, d=32, layers=2, window=128, num_heads=2, topk=4)
    for length in (16, 32, 64, 128):
        x = torch.randint(0, 64, (2, length))
        logits = model(x)
        assert logits.shape == (2, 64)
        assert torch.isfinite(logits).all()


def test_pc_sts_v3_param_count():
    model = PCSTSLMv3(vocab=512, d=192, layers=8, window=256, num_heads=4)
    params = count_params(model)
    assert params > 0
    print(f"v3 model params: {params:,}")
