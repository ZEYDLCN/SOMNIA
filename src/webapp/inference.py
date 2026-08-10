"""
Web formu için gerçek model çıkarımı — ONNX Runtime ile.

Kullanıcı en az `window_size` (7) günlük veri girdiğinde, eğitilmiş
Temporal Transformer'ın (bkz. src/models/export_onnx.py) son N gününü
kullanarak bir sonraki gecenin tahmini uyku kalitesini hesaplar.

Neden ONNX Runtime (torch değil): tam PyTorch yüzlerce MB'tır ve
PythonAnywhere/Koyeb gibi ücretsiz platformların disk kotasını
zorlayabilir. ONNX Runtime ~50-60MB'tır ve sadece çıkarım için
yeterlidir — burada eğitim yapılmıyor.

ÖNEMLİ (dürüstçe belirtiyoruz): Model sentetik veri üzerinde eğitildi.
Gerçek bir kullanıcının verisine uygulamak bir demo/gösterge niteliğindedir,
kesin bir tıbbi/bilimsel tahmin değildir — bkz. docs/plan.md §5, §9.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.data.preprocessing import (
    DATE_COL,
    TARGET_COL,
    engineer_features,
    handle_missing,
)

MODEL_DIR = Path(__file__).resolve().parent / "model"
ONNX_PATH = MODEL_DIR / "temporal_transformer.onnx"
NORMALIZER_PATH = MODEL_DIR / "normalizer.json"

_session: ort.InferenceSession | None = None
_normalizer: dict | None = None


def _load() -> tuple[ort.InferenceSession, dict]:
    global _session, _normalizer
    if _session is None:
        _session = ort.InferenceSession(str(ONNX_PATH))
        _normalizer = json.loads(NORMALIZER_PATH.read_text())
    return _session, _normalizer


def window_size() -> int:
    _, normalizer = _load()
    return normalizer["window_size"]


def entries_to_dataframe(entries: list) -> pd.DataFrame:
    """sqlite3.Row listesini (tarihe göre ARTAN sırada), preprocessing.py
    ile aynı şemaya (day_of_week dahil) sahip bir DataFrame'e çevirir."""
    rows = []
    for e in entries:
        d = pd.Timestamp(e["date"])
        rows.append(
            {
                DATE_COL: d,
                "day_of_week": d.weekday(),
                "stress_score": e["stress_score"],
                "caffeine_mg": e["caffeine_mg"],
                "caffeine_hours_before_bed": e["caffeine_hours_before_bed"],
                "screen_time_before_bed_min": e["screen_time_before_bed_min"],
                "room_temp_c": e["room_temp_c"],
                "noise_level_db": e["noise_level_db"],
                "exercise_minutes": e["exercise_minutes"],
                "exercise_time_of_day": e["exercise_time_of_day"],
                "last_meal_delta_hr": e["last_meal_delta_hr"],
                "sleep_duration_min": e["sleep_duration_min"],
                TARGET_COL: e["sleep_quality"],
            }
        )
    df = pd.DataFrame(rows).sort_values(DATE_COL).reset_index(drop=True)
    return df


def predict_next_sleep_quality(entries: list) -> float:
    """`entries`: kullanıcının en az window_size günlük en güncel
    kayıtları (herhangi bir sırada; burada tarihe göre sıralanır ve son
    window_size günü alınır). Bir sonraki gecenin tahmini sleep_quality
    değerini (1-10 skalasında, klipsiz ham model çıktısı) döndürür."""
    session, normalizer = _load()
    w = normalizer["window_size"]

    df = entries_to_dataframe(entries)
    if len(df) < w:
        raise ValueError(f"En az {w} günlük kayıt gerekli, {len(df)} var.")
    df = df.tail(w).reset_index(drop=True)

    df = handle_missing(df)
    df, feature_cols = engineer_features(df)

    # Eğitimdeki özellik SIRASIYLA birebir aynı olmalı.
    expected = normalizer["feature_names"]
    if feature_cols != expected:
        # Sıra farklıysa yeniden diz (kolon isimleri aynı, sıra garantisi için).
        df = df[expected + [c for c in df.columns if c not in expected]]
        feature_cols = expected

    values = df[feature_cols].to_numpy(dtype=np.float32)  # (w, n_features)
    mean = np.array(normalizer["mean"], dtype=np.float32)
    std = np.array(normalizer["std"], dtype=np.float32)
    normalized = (values - mean) / std

    x = normalized.reshape(1, w, -1).astype(np.float32)
    pred = session.run(None, {"window": x})[0]
    return float(pred.reshape(-1)[0])
