"""
Yorumlanabilirlik modülü — SOMNIA (Kişisel Uyku Kalitesi için Temporal Transformer)

Adım 6 (bkz. docs/plan.md §5): attention haritaları + permutation
importance ile çapraz doğrulama.

ÖNEMLİ (plan.md §5.2): Attention ağırlıkları TEK BAŞINA nedensellik kanıtı
DEĞİLDİR. Burada iki farklı yöntemi (attention vs. permutation importance)
karşılaştırarak, en azından "modelin neye baktığı" ile "modelin performansı
için neyin gerçekten gerekli olduğu" arasında bir tutarlılık kontrolü
yapıyoruz. İkisinin de aynı yönü işaret etmesi güven artırır ama yine de
KORELASYONdur — nedensellik için adım 7'deki Granger testi ve confounder
tartışması gerekir.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import build_pipeline, WindowedDataset
from src.models.transformer import TemporalSleepTransformer, TransformerConfig

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "temporal_transformer.pt"


def load_model() -> tuple[TemporalSleepTransformer, TransformerConfig]:
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = TransformerConfig(**ckpt["config"])
    model = TemporalSleepTransformer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def collect_attention(model: TemporalSleepTransformer, ds: WindowedDataset) -> dict:
    x = torch.tensor(ds.X, dtype=torch.float32)
    _, attn = model(x, return_attn=True)
    return {
        "feature_attn": attn["feature_attn"].numpy(),  # (N, T, F)
        "temporal_pool_attn": attn["temporal_pool_attn"].numpy(),  # (N, T)
        "temporal_layer_attn": [a.numpy() for a in attn["temporal_layer_attn"]],
    }


def feature_importance_from_attention(
    feature_attn: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """(N, T, F) attention ağırlıklarını örnekler ve günler üzerinden
    ortalayarak, her özelliğe tek bir 'ortalama attention' skoru verir."""
    mean_per_feature = feature_attn.mean(axis=(0, 1))  # (F,)
    return dict(zip(feature_names, mean_per_feature.tolist()))


def day_importance_from_attention(temporal_pool_attn: np.ndarray) -> dict[str, float]:
    """(N, T) -> her göreli gün konumu için ortalama attention.
    t-T (en eski) .. t-1 (en yeni) etiketleriyle."""
    T = temporal_pool_attn.shape[1]
    mean_per_day = temporal_pool_attn.mean(axis=0)  # (T,)
    labels = [f"t-{T - i}" for i in range(T)]
    return dict(zip(labels, mean_per_day.tolist()))


@torch.no_grad()
def _mae(model: TemporalSleepTransformer, X: np.ndarray, y: np.ndarray) -> float:
    pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return float(np.mean(np.abs(pred - y)))


def permutation_importance(
    model: TemporalSleepTransformer,
    ds: WindowedDataset,
    n_repeats: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Her özellik sütununu (tüm günler boyunca birlikte) rastgele karıştırıp
    val MAE'deki artışı ölçer. Artış ne kadar büyükse, model performansı o
    özelliğe o kadar bağımlı demektir — bu, attention'ı çapraz doğrulamak
    için ayrı, bağımsız bir yöntemdir (plan.md §5.2)."""
    rng = np.random.default_rng(seed)
    baseline_mae = _mae(model, ds.X, ds.y)

    importances: dict[str, float] = {}
    n_features = ds.X.shape[-1]
    for f_idx, name in enumerate(ds.feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = ds.X.copy()
            perm = rng.permutation(X_perm.shape[0])
            X_perm[:, :, f_idx] = X_perm[perm, :, f_idx]
            mae = _mae(model, X_perm, ds.y)
            deltas.append(mae - baseline_mae)
        importances[name] = float(np.mean(deltas))

    return {"baseline_mae": baseline_mae, "per_feature": importances}


def _rank(d: dict[str, float]) -> dict[str, int]:
    """Büyükten küçüğe sıralayıp 1'den başlayan rank (1 = en önemli) verir."""
    ordered = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    return {name: rank + 1 for rank, (name, _) in enumerate(ordered)}


def _spearman(rank_a: dict[str, int], rank_b: dict[str, int]) -> float:
    names = list(rank_a)
    a = np.array([rank_a[n] for n in names], dtype=np.float64)
    b = np.array([rank_b[n] for n in names], dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def plot_feature_attention_heatmap(feature_attn: np.ndarray, feature_names: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = feature_attn.shape[1]
    heatmap = feature_attn.mean(axis=0)  # (T, F)
    day_labels = [f"t-{T - i}" for i in range(T)]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(T))
    ax.set_yticklabels(day_labels)
    ax.set_xlabel("özellik")
    ax.set_ylabel("geçmiş gün")
    ax.set_title("Özellik-ekseni Attention Isı Haritası (test seti ortalaması)")
    fig.colorbar(im, ax=ax, label="attention ağırlığı")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_day_importance(day_importance: dict[str, float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(day_importance.keys())
    values = list(day_importance.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="tab:blue")
    ax.set_xlabel("geçmiş gün (t = tahmin edilen gece)")
    ax.set_ylabel("ortalama attention ağırlığı")
    ax.set_title("Zaman-ekseni Attention: Hangi Geçmiş Gün Daha Etkili?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attention_vs_permutation(
    attn_importance: dict[str, float],
    perm_importance: dict[str, float],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(attn_importance.keys())
    attn_vals = np.array([attn_importance[n] for n in names])
    perm_vals = np.array([perm_importance[n] for n in names])

    # Her ikisini de 0-1 aralığına ölçekleyip yan yana karşılaştır.
    def _norm(v):
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)

    attn_n = _norm(attn_vals)
    perm_n = _norm(perm_vals)

    order = np.argsort(-attn_n)
    names_sorted = [names[i] for i in order]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, attn_n[order], width, label="attention (normalize)")
    ax.bar(x + width / 2, perm_n[order], width, label="permutation importance (normalize)")
    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("normalize önem skoru")
    ax.set_title("Attention vs. Permutation Importance — Çapraz Doğrulama")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    model, cfg = load_model()
    splits = build_pipeline(window_size=cfg.window_size)

    # Attention haritaları: test seti üzerinde (görülmemiş veri).
    attn = collect_attention(model, splits.test)
    feat_importance = feature_importance_from_attention(attn["feature_attn"], splits.test.feature_names)
    day_importance = day_importance_from_attention(attn["temporal_pool_attn"])

    # Permutation importance: aynı test seti üzerinde, bağımsız yöntem.
    perm = permutation_importance(model, splits.test)

    attn_rank = _rank(feat_importance)
    perm_rank = _rank(perm["per_feature"])
    agreement = _spearman(attn_rank, perm_rank)

    print("Özellik önemi (attention, azalan sırada):")
    for name, _ in sorted(feat_importance.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:30s} attn={feat_importance[name]:.4f}  perm_delta_mae={perm['per_feature'][name]:+.4f}")

    print(f"\nAttention-rank vs. permutation-rank Spearman korelasyonu: {agreement:.3f}")
    print(
        "(1.0'a yakın = iki bağımsız yöntem aynı özellikleri önemli buluyor; "
        "yine de bu KORELASYONDUR, nedensellik kanıtı değildir — bkz. docs/plan.md §5)"
    )

    print("\nGün önemi (attention, t=tahmin edilen gece):")
    for label, val in day_importance.items():
        print(f"  {label}: {val:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_attention_heatmap(
        attn["feature_attn"], splits.test.feature_names, REPORTS_DIR / "attention_feature_heatmap.png"
    )
    plot_day_importance(day_importance, REPORTS_DIR / "attention_day_importance.png")
    plot_attention_vs_permutation(
        feat_importance, perm["per_feature"], REPORTS_DIR / "attention_vs_permutation.png"
    )

    with open(REPORTS_DIR / "interpretability_results.json", "w") as f:
        json.dump(
            {
                "feature_importance_attention": feat_importance,
                "day_importance_attention": day_importance,
                "permutation_importance": perm,
                "attention_vs_permutation_spearman": agreement,
                "note": (
                    "Attention ve permutation importance birbirini bağımsız şekilde "
                    "doğrular ama ikisi de korelasyon/duyarlılık ölçümüdür, nedensellik "
                    "kanıtı değildir. Nedensellik tartışması için docs/plan.md §5 ve "
                    "adım 7'deki Granger causality analizine bakınız."
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nKaydedildi: {REPORTS_DIR}/attention_feature_heatmap.png")
    print(f"Kaydedildi: {REPORTS_DIR}/attention_day_importance.png")
    print(f"Kaydedildi: {REPORTS_DIR}/attention_vs_permutation.png")
    print(f"Kaydedildi: {REPORTS_DIR}/interpretability_results.json")


if __name__ == "__main__":
    main()
