import torch

from src.models.transformer import TemporalSleepTransformer, TransformerConfig


def _make_model(n_features: int = 16, window_size: int = 7) -> TemporalSleepTransformer:
    cfg = TransformerConfig(
        n_features=n_features,
        window_size=window_size,
        d_model=8,
        n_heads=2,
        n_temporal_layers=1,
        dropout=0.0,
    )
    return TemporalSleepTransformer(cfg)


def test_forward_shape():
    model = _make_model()
    x = torch.randn(4, 7, 16)
    pred = model(x)
    assert pred.shape == (4,)


def test_attention_shapes_and_normalization():
    model = _make_model()
    x = torch.randn(3, 7, 16)
    pred, attn = model(x, return_attn=True)

    assert pred.shape == (3,)
    assert attn["feature_attn"].shape == (3, 7, 16)
    assert attn["temporal_pool_attn"].shape == (3, 7)
    assert len(attn["temporal_layer_attn"]) == 1
    assert attn["temporal_layer_attn"][0].shape == (3, 7, 7)

    # Attention ağırlıkları bir softmax çıktısıdır: satır toplamları ~1.
    feat_sums = attn["feature_attn"].sum(dim=-1)
    assert torch.allclose(feat_sums, torch.ones_like(feat_sums), atol=1e-4)

    pool_sums = attn["temporal_pool_attn"].sum(dim=-1)
    assert torch.allclose(pool_sums, torch.ones_like(pool_sums), atol=1e-4)


def test_gradients_flow():
    model = _make_model()
    x = torch.randn(4, 7, 16, requires_grad=False)
    y = torch.randn(4)
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # NaN kontrolü
