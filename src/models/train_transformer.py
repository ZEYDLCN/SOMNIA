"""
Eğitim döngüsü — Temporal Transformer (SOMNIA)

Adım 4-5 (bkz. docs/plan.md): modeli walk-forward train/val split'i ile
eğitir, en iyi val MAE'ye sahip checkpoint'i saklar (early stopping), ve
test setinde son bir kez değerlendirir. Sonuçları baseline'larla
(reports/baseline_results.json) karşılaştırır.

Küçük veri seti (205 train örneği) nedeniyle model kasıtlı olarak küçük
tutuldu (d_model=32, 2 katman) ve dropout ile regularize edildi — plan.md
§9'daki "küçük veri setinde Transformer overkill olabilir" riskine karşı.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocessing import build_pipeline, WindowedDataset
from src.models.transformer import TemporalSleepTransformer, TransformerConfig, count_parameters

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

SEED = 42


def _set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _to_loader(ds: WindowedDataset, batch_size: int, shuffle: bool) -> DataLoader:
    X = torch.tensor(ds.X, dtype=torch.float32)
    y = torch.tensor(ds.y, dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        preds.append(model(xb).numpy())
        targets.append(yb.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    return {"mae": mae, "rmse": rmse}


def train(
    window_size: int = 7,
    d_model: int = 32,
    n_heads: int = 4,
    n_temporal_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 300,
    patience: int = 30,
) -> dict:
    _set_seed()
    splits = build_pipeline(window_size=window_size)
    n_features = len(splits.train.feature_names)

    cfg = TransformerConfig(
        n_features=n_features,
        window_size=window_size,
        d_model=d_model,
        n_heads=n_heads,
        n_temporal_layers=n_temporal_layers,
        dropout=dropout,
    )
    model = TemporalSleepTransformer(cfg)
    print(f"Model parametre sayısı: {count_parameters(model):,}")

    train_loader = _to_loader(splits.train, batch_size, shuffle=True)
    val_loader = _to_loader(splits.val, batch_size, shuffle=False)
    test_loader = _to_loader(splits.test, batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()  # Huber — öz-bildirim hedeflerindeki gürültüye dayanıklı

    best_val_mae = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_mae": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_metrics = _evaluate(model, val_loader)
        history["train_loss"].append(epoch_loss / n_batches)
        history["val_mae"].append(val_metrics["mae"])

        if val_metrics["mae"] < best_val_mae - 1e-4:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d} | train_loss {history['train_loss'][-1]:.4f} "
                f"| val_MAE {val_metrics['mae']:.4f} | best_val_MAE {best_val_mae:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping @ epoch {epoch} (patience={patience})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    val_metrics = _evaluate(model, val_loader)
    test_metrics = _evaluate(model, test_loader)

    print(f"\nEn iyi model -> val_MAE: {val_metrics['mae']:.4f}, val_RMSE: {val_metrics['rmse']:.4f}")
    print(f"En iyi model -> test_MAE: {test_metrics['mae']:.4f}, test_RMSE: {test_metrics['rmse']:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": best_state, "config": cfg.__dict__},
        CHECKPOINT_DIR / "temporal_transformer.pt",
    )

    results = {
        "config": cfg.__dict__,
        "n_parameters": count_parameters(model),
        "epochs_trained": len(history["train_loss"]),
        "val": val_metrics,
        "test": test_metrics,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "transformer_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    _plot_history(history)
    _compare_to_baselines(results)

    return results


def _plot_history(history: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(history["train_loss"], label="train loss (Huber)", color="tab:blue")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(history["val_mae"], label="val MAE", color="tab:orange")
    ax2.set_ylabel("val MAE", color="tab:orange")

    fig.suptitle("Temporal Transformer Eğitim Eğrisi")
    fig.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_DIR / "transformer_training_curve.png", dpi=150)
    plt.close(fig)


def _compare_to_baselines(results: dict) -> None:
    baseline_path = REPORTS_DIR / "baseline_results.json"
    if not baseline_path.exists():
        return
    baselines = json.loads(baseline_path.read_text())

    all_results = dict(baselines)
    all_results["temporal_transformer"] = {
        "val": results["val"],
        "test": results["test"],
    }

    header = f"{'model':22s} {'val_MAE':>8s} {'val_RMSE':>9s} {'test_MAE':>9s} {'test_RMSE':>10s}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in all_results.items():
        print(
            f"{name:22s} {m['val']['mae']:8.3f} {m['val']['rmse']:9.3f} "
            f"{m['test']['mae']:9.3f} {m['test']['rmse']:10.3f}"
        )

    with open(REPORTS_DIR / "all_model_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    train()
