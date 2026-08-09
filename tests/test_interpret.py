import numpy as np
import pytest

from src.data.preprocessing import build_pipeline
from src.models.interpret import (
    CHECKPOINT_PATH,
    collect_attention,
    day_importance_from_attention,
    feature_importance_from_attention,
    load_model,
    permutation_importance,
)

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="Önce 'python -m src.models.train_transformer' çalıştırılıp checkpoint üretilmeli.",
)


def test_attention_and_permutation_importance_are_consistent_shapes():
    model, cfg = load_model()
    splits = build_pipeline(window_size=cfg.window_size)

    attn = collect_attention(model, splits.test)
    n, T, F = attn["feature_attn"].shape
    assert (n, T, F) == splits.test.X.shape
    assert attn["temporal_pool_attn"].shape == (n, T)

    feat_imp = feature_importance_from_attention(attn["feature_attn"], splits.test.feature_names)
    assert set(feat_imp) == set(splits.test.feature_names)
    # her özelliğin ortalama attention'ı [0, 1] aralığında olmalı (softmax çıktısı)
    assert all(0.0 <= v <= 1.0 for v in feat_imp.values())
    # bütün özellikler üzerinden toplam ~1 olmalı (her gün için softmax normalize)
    assert abs(sum(feat_imp.values()) - 1.0) < 1e-3

    day_imp = day_importance_from_attention(attn["temporal_pool_attn"])
    assert len(day_imp) == T
    assert abs(sum(day_imp.values()) - 1.0) < 1e-3


def test_permutation_importance_runs():
    model, cfg = load_model()
    splits = build_pipeline(window_size=cfg.window_size)
    perm = permutation_importance(model, splits.test, n_repeats=3)

    assert set(perm["per_feature"]) == set(splits.test.feature_names)
    assert perm["baseline_mae"] > 0
    assert all(np.isfinite(v) for v in perm["per_feature"].values())
