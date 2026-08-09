"""
Baseline modeller — SOMNIA (Kişisel Uyku Kalitesi için Temporal Transformer)

Adım 3 (bkz. docs/plan.md §6-7): Transformer'ın gerçekten değer katıp
katmadığını dürüstçe göstermek için önce üç basit baseline kuruyoruz:

    1. Naive        — "dünkü değeri tekrar et" (pencerenin son günündeki
                        sleep_quality'yi bir sonraki gecenin tahmini say)
    2. Linear Reg.   — düzleştirilmiş pencere (window_size * n_features)
                        üzerinde doğrusal regresyon
    3. XGBoost       — aynı düzleştirilmiş girdi üzerinde gradient boosting

Değerlendirme walk-forward split'in val/test kısımlarında, MAE ve RMSE ile.
Sonuçlar reports/baseline_results.json içine yazılır.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.data.preprocessing import (
    TARGET_COL,
    RAW_CSV,
    WindowedDataset,
    build_pipeline,
    engineer_features,
    handle_missing,
    load_raw,
    make_windows,
    walk_forward_split,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def naive_predictions(ds_raw: WindowedDataset) -> np.ndarray:
    """Pencerenin son gününe ait (ham, normalize edilmemiş) sleep_quality
    değerini bir sonraki gecenin tahmini olarak kullanır."""
    target_idx = ds_raw.feature_names.index(TARGET_COL)
    return ds_raw.X[:, -1, target_idx]


def flatten(ds: WindowedDataset) -> np.ndarray:
    return ds.X.reshape(ds.X.shape[0], -1)


def run(window_size: int = 7) -> dict:
    # Normalize edilmiş (model girdisi) ve ham (naive baseline için) iki
    # ayrı pencereleme üretiyoruz; aynı window_size/split oranlarıyla
    # kesişme noktaları birebir aynı, dolayısıyla y dizileri de eşleşir.
    splits_norm = build_pipeline(window_size=window_size)

    df = handle_missing(load_raw(RAW_CSV))
    df, feature_cols = engineer_features(df)
    windowed_raw = make_windows(df, feature_cols, window_size)
    train_raw, val_raw, test_raw = walk_forward_split(windowed_raw)

    assert np.array_equal(train_raw.y, splits_norm.train.y)
    assert np.array_equal(val_raw.y, splits_norm.val.y)
    assert np.array_equal(test_raw.y, splits_norm.test.y)

    results: dict[str, dict[str, dict[str, float]]] = {}

    # 1) Naive
    naive_val = naive_predictions(val_raw)
    naive_test = naive_predictions(test_raw)
    results["naive"] = {
        "val": _metrics(val_raw.y, naive_val),
        "test": _metrics(test_raw.y, naive_test),
    }

    # 2) Linear Regression (düzleştirilmiş, normalize edilmiş pencere)
    X_train, y_train = flatten(splits_norm.train), splits_norm.train.y
    X_val, y_val = flatten(splits_norm.val), splits_norm.val.y
    X_test, y_test = flatten(splits_norm.test), splits_norm.test.y

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    results["linear_regression"] = {
        "val": _metrics(y_val, lin.predict(X_val)),
        "test": _metrics(y_test, lin.predict(X_test)),
    }

    # 3) XGBoost
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    results["xgboost"] = {
        "val": _metrics(y_val, xgb.predict(X_val)),
        "test": _metrics(y_test, xgb.predict(X_test)),
    }

    return results


def print_table(results: dict) -> None:
    header = f"{'model':20s} {'val_MAE':>8s} {'val_RMSE':>9s} {'test_MAE':>9s} {'test_RMSE':>10s}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:20s} {m['val']['mae']:8.3f} {m['val']['rmse']:9.3f} "
            f"{m['test']['mae']:9.3f} {m['test']['rmse']:10.3f}"
        )


def save_results(results: dict, out_dir: Path = REPORTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return out_path


def plot_comparison(results: dict, out_dir: Path = REPORTS_DIR) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results.keys())
    val_mae = [results[m]["val"]["mae"] for m in models]
    test_mae = [results[m]["test"]["mae"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, val_mae, width, label="val MAE")
    ax.bar(x + width / 2, test_mae, width, label="test MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("MAE (sleep_quality, 1-10)")
    ax.set_title("Baseline Karşılaştırması")
    ax.legend()
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_mae_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    results = run()
    print_table(results)
    json_path = save_results(results)
    png_path = plot_comparison(results)
    print(f"\nKaydedildi: {json_path}")
    print(f"Kaydedildi: {png_path}")


if __name__ == "__main__":
    main()
