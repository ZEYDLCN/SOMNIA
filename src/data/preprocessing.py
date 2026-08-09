"""
Veri ön işleme pipeline'ı — SOMNIA (Kişisel Uyku Kalitesi için Temporal Transformer)

Adım 2 (bkz. docs/plan.md): normalizasyon, windowing, eksik veri.

Akış:
    ham CSV
      -> temizleme (eksik veri doldurma)
      -> feature engineering (döngüsel gün encoding, kategorik encode)
      -> zaman sırasına saygılı train/val/test split (shuffle YOK)
      -> normalizasyon (yalnızca train üzerinde fit edilir — leakage yok)
      -> windowing: [gün t-N, ..., gün t-1] -> hedef sleep_quality(gün t)

Not: `data/ground_truth.json` sadece sentetik veri üretiminde gömülen
nedensel katsayıları belgeler; bu modül veya sonraki model eğitimi onu
GİRDİ olarak kullanmaz — yalnızca sonuçları doğrulamak için ayrı bir
adımda referans alınacaktır.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_CSV = DATA_DIR / "synthetic_sleep_data.csv"
PROCESSED_DIR = DATA_DIR / "processed"

TARGET_COL = "sleep_quality"
DATE_COL = "date"
DOW_COL = "day_of_week"

# Ham CSV'deki, pencere içine "geçmiş özellik" olarak giren sütunlar.
# sleep_quality kasıtlı olarak dahil: day_persistence (ground truth'ta 0.3)
# gibi otoregresif etkiyi modelin geçmiş gecelerden öğrenebilmesi için.
NUMERIC_FEATURE_COLS = [
    "stress_score",
    "caffeine_mg",
    "caffeine_hours_before_bed",
    "screen_time_before_bed_min",
    "room_temp_c",
    "noise_level_db",
    "exercise_minutes",
    "last_meal_delta_hr",
    "sleep_duration_min",
    TARGET_COL,
]
CATEGORICAL_FEATURE_COLS = ["exercise_time_of_day"]
EXERCISE_TIME_CATEGORIES = ["none", "morning", "afternoon", "evening"]


@dataclasses.dataclass
class WindowedDataset:
    X: np.ndarray  # (n_samples, window_size, n_features)
    y: np.ndarray  # (n_samples,)
    dates: np.ndarray  # (n_samples,) hedef günün tarihi
    feature_names: list[str]


@dataclasses.dataclass
class SplitDatasets:
    train: WindowedDataset
    val: WindowedDataset
    test: WindowedDataset
    feature_means: np.ndarray
    feature_stds: np.ndarray


def load_raw(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik veriyi doldurur: zaman sırasına göre interpolasyon,
    kalan uçlarda forward/backward-fill, kategorik sütunlarda 'none'."""
    df = df.copy()
    for col in NUMERIC_FEATURE_COLS:
        if df[col].isna().any():
            df[col] = df[col].interpolate(limit_direction="both")
    for col in CATEGORICAL_FEATURE_COLS:
        df[col] = df[col].fillna("none")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Döngüsel gün-of-week encoding + kategorik one-hot ekler.
    Geriye özellik matrisi (sleep_quality dahil, date/day_of_week hariç
    ham haliyle) ve sütun sırası döner."""
    df = df.copy()

    # Haftanın günü döngüsel (cyclical) encoding: 7 günlük periyot.
    df["dow_sin"] = np.sin(2 * np.pi * df[DOW_COL] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df[DOW_COL] / 7)

    # Kategorik: egzersiz zamanı -> one-hot (sabit kategori sırası).
    for cat in EXERCISE_TIME_CATEGORIES:
        df[f"exercise_time_{cat}"] = (df["exercise_time_of_day"] == cat).astype(float)

    feature_cols = (
        NUMERIC_FEATURE_COLS
        + ["dow_sin", "dow_cos"]
        + [f"exercise_time_{cat}" for cat in EXERCISE_TIME_CATEGORIES]
    )
    return df, feature_cols


def make_windows(
    df: pd.DataFrame, feature_cols: list[str], window_size: int
) -> WindowedDataset:
    """[t-window_size, ..., t-1] özelliklerinden gün t'nin sleep_quality
    değerini tahmin edecek şekilde kayan pencereler üretir. Hedef günün
    kendi özellikleri pencereye DAHİL DEĞİLDİR (plan.md §4 ile tutarlı:
    "sonraki gecenin" tahmini, o gecenin kendi verisi olmadan yapılır)."""
    values = df[feature_cols].to_numpy(dtype=np.float64)
    target = df[TARGET_COL].to_numpy(dtype=np.float64)
    dates = df[DATE_COL].to_numpy()

    n = len(df)
    X, y, y_dates = [], [], []
    for end in range(window_size, n):
        X.append(values[end - window_size : end])
        y.append(target[end])
        y_dates.append(dates[end])

    return WindowedDataset(
        X=np.stack(X) if X else np.empty((0, window_size, len(feature_cols))),
        y=np.array(y, dtype=np.float64),
        dates=np.array(y_dates),
        feature_names=feature_cols,
    )


def walk_forward_split(
    windowed: WindowedDataset,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
    """Zaman sırasına saygılı bölme — shuffle YOK. Örnekler zaten
    tarihe göre sıralı olduğundan, sadece indeks üzerinden kesilir."""
    n = len(windowed.y)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    def _slice(lo: int, hi: int) -> WindowedDataset:
        return WindowedDataset(
            X=windowed.X[lo:hi],
            y=windowed.y[lo:hi],
            dates=windowed.dates[lo:hi],
            feature_names=windowed.feature_names,
        )

    train = _slice(0, n_train)
    val = _slice(n_train, n_train + n_val)
    test = _slice(n_train + n_val, n)
    return train, val, test


def fit_normalizer(train: WindowedDataset) -> tuple[np.ndarray, np.ndarray]:
    """Yalnızca train penceresindeki özellik değerlerinden mean/std
    hesaplar (data leakage önlenir). Şekil: (n_features,)."""
    flat = train.X.reshape(-1, train.X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0  # sabit sütunlarda bölme hatasını önle
    return mean, std


def apply_normalizer(
    windowed: WindowedDataset, mean: np.ndarray, std: np.ndarray
) -> WindowedDataset:
    return WindowedDataset(
        X=(windowed.X - mean) / std,
        y=windowed.y,
        dates=windowed.dates,
        feature_names=windowed.feature_names,
    )


def build_pipeline(
    csv_path: Path = RAW_CSV,
    window_size: int = 7,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitDatasets:
    df = load_raw(csv_path)
    df = handle_missing(df)
    df, feature_cols = engineer_features(df)

    windowed = make_windows(df, feature_cols, window_size)
    train, val, test = walk_forward_split(windowed, train_frac, val_frac)

    mean, std = fit_normalizer(train)
    train = apply_normalizer(train, mean, std)
    val = apply_normalizer(val, mean, std)
    test = apply_normalizer(test, mean, std)

    return SplitDatasets(
        train=train, val=val, test=test, feature_means=mean, feature_stds=std
    )


def save_processed(splits: SplitDatasets, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ds in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        np.savez(
            out_dir / f"{name}.npz",
            X=ds.X,
            y=ds.y,
            dates=ds.dates.astype(str),
            feature_names=np.array(ds.feature_names),
        )
    np.savez(
        out_dir / "normalizer.npz",
        mean=splits.feature_means,
        std=splits.feature_stds,
    )


def main() -> None:
    splits = build_pipeline()
    print(f"Özellikler ({len(splits.train.feature_names)}): {splits.train.feature_names}")
    for name, ds in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        print(
            f"{name:5s} -> X: {ds.X.shape}, y: {ds.y.shape}, "
            f"tarih aralığı: {ds.dates.min()} .. {ds.dates.max()}"
        )
    save_processed(splits)
    print(f"\nKaydedildi: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
