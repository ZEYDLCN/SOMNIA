import pytest

from src.data.preprocessing import RAW_CSV, load_raw
from src.webapp.inference import ONNX_PATH, predict_next_sleep_quality, window_size

pytestmark = pytest.mark.skipif(
    not ONNX_PATH.exists(),
    reason="Önce 'python -m src.models.export_onnx' çalıştırılıp model export edilmeli.",
)


def _sample_entries(n: int) -> list[dict]:
    df = load_raw(RAW_CSV).tail(n)
    return df.to_dict("records")


def test_window_size_matches_training_config():
    assert window_size() == 7


def test_predict_returns_plausible_sleep_quality():
    entries = _sample_entries(window_size())
    pred = predict_next_sleep_quality(entries)
    assert isinstance(pred, float)
    # Model 1-10 skalasında eğitildi; makul aralıkta olmalı (kesin
    # klipslenmemiş çıktı, biraz taşabilir ama patlamamalı).
    assert -2.0 < pred < 14.0


def test_predict_requires_minimum_entries():
    entries = _sample_entries(window_size() - 1)
    with pytest.raises(ValueError):
        predict_next_sleep_quality(entries)


def test_predict_is_deterministic():
    entries = _sample_entries(window_size())
    pred1 = predict_next_sleep_quality(entries)
    pred2 = predict_next_sleep_quality(entries)
    assert pred1 == pred2
