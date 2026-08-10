"""
Sonuç raporu / dashboard üretici — SOMNIA (Kişisel Uyku Kalitesi için
Temporal Transformer)

Adım 8 (bkz. docs/plan.md): reports/*.json içindeki gerçek sonuçları okuyup
tek sayfalık, kendi kendine yeten (self-contained) bir HTML dashboard
üretir. Veri her zaman JSON dosyalarından okunur — sayılar koda gömülü
değildir, rapor tekrar çalıştırıldığında güncel sonuçları yansıtır.

Kullanım:
    python -m src.report.build_report
    -> docs/index.html
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
OUT_PATH = ROOT / "docs" / "index.html"

# Bu statik rapor (GitHub Pages) ve veri girişi formu (Flask, PythonAnywhere'de
# barındırılıyor — bkz. DEPLOY.md) İKİ AYRI deployment; bu link ikisini
# birbirine bağlayan tek yol.
WEBAPP_URL = "https://zeydalcan.pythonanywhere.com"

# Marka sembolü ("Arc Wave" S) — bkz. docs/assets/. Tema tokenlarına
# bağlı (currentColor/var(--c-accent)) ki açık/koyu temada otomatik uyarlansın.
LOGO_SYMBOL_PATH = "M33 11 C36 3 10 2 11 14 C11 24 33 32 33 42 C34 54 10 54 11 45"
LOGO_SYMBOL_NODES = [(22, 5, 1.0), (22, 28, 0.68), (22, 51, 1.0)]


def logo_symbol_svg(h: float, mist: bool = False) -> str:
    """mist=False (varsayılan): tema tokenlarına bağlı (var(--ink)/var(--c-accent)),
    sayfa içine gömülen kullanımlar için (açık/koyu temaya uyarlanır).
    mist=True: sabit Mist/Lavender renkleri — splash ekranı gibi, her zaman
    aynı (midnight arka planlı) marka anını temsil eden, temadan bağımsız
    kullanımlar için."""
    sw = max(1.5, h * 0.058)
    w = round(h * 44 / 56, 2)
    stroke_color = "#E8E6F5" if mist else "var(--ink)"
    node_color = "#8B85C8" if mist else "var(--c-accent)"
    circles = "\n".join(
        f'<circle cx="{cx}" cy="{cy}" r="{sw * 0.88 * mult:.2f}" fill="{node_color}"/>'
        for cx, cy, mult in LOGO_SYMBOL_NODES
    )
    return (
        f'<svg viewBox="0 0 44 56" width="{w}" height="{h}" fill="none" '
        f'style="overflow:visible; flex-shrink:0;" aria-hidden="true">'
        f'<path d="{LOGO_SYMBOL_PATH}" stroke="{stroke_color}" stroke-width="{sw:.2f}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>{circles}</svg>'
    )

MODEL_LABELS = {
    "naive": "Naive",
    "linear_regression": "Linear Regression",
    "xgboost": "XGBoost",
    "temporal_transformer": "Temporal Transformer",
}
MODEL_COLOR_VAR = {
    "naive": "var(--c-model-naive)",
    "linear_regression": "var(--c-model-linear)",
    "xgboost": "var(--c-model-xgb)",
    "temporal_transformer": "var(--c-model-tft)",
}

FEATURE_LABELS = {
    "stress_score": "Stres skoru",
    "caffeine_mg": "Kafein (mg)",
    "caffeine_hours_before_bed": "Kafein → yatış (saat)",
    "screen_time_before_bed_min": "Ekran süresi (yatmadan önce)",
    "room_temp_c": "Oda sıcaklığı",
    "noise_level_db": "Gürültü (dB)",
    "exercise_minutes": "Egzersiz süresi",
    "last_meal_delta_hr": "Son yemek → yatış",
    "sleep_duration_min": "Uyku süresi (geçmiş)",
    "sleep_quality": "Uyku kalitesi (geçmiş gece)",
    "dow_sin": "Haftanın günü (sin)",
    "dow_cos": "Haftanın günü (cos)",
    "exercise_time_none": "Egzersiz yok",
    "exercise_time_morning": "Egzersiz: sabah",
    "exercise_time_afternoon": "Egzersiz: öğleden sonra",
    "exercise_time_evening": "Egzersiz: akşam",
}


def _load(name: str) -> dict:
    return json.loads((REPORTS_DIR / name).read_text())


def esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# SVG yardımcıları
# ---------------------------------------------------------------------------


def model_comparison_chart(baselines: dict, transformer: dict) -> str:
    order = ["naive", "linear_regression", "xgboost", "temporal_transformer"]
    test_mae = {
        **{k: v["test"]["mae"] for k, v in baselines.items()},
        "temporal_transformer": transformer["test"]["mae"],
    }
    val_mae = {
        **{k: v["val"]["mae"] for k, v in baselines.items()},
        "temporal_transformer": transformer["val"]["mae"],
    }

    max_val = max(max(test_mae.values()), max(val_mae.values())) * 1.12
    W, H = 640, 300
    pad_l, pad_r, pad_t, pad_b = 8, 8, 12, 34
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    group_w = plot_w / len(order)
    bar_w = group_w * 0.32
    gap = group_w * 0.06

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1 - v / max_val)

    bars = []
    gridlines = []
    n_grid = 4
    for i in range(n_grid + 1):
        gv = max_val * i / n_grid
        gy = y_of(gv)
        gridlines.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{W - pad_r}" y2="{gy:.1f}" '
            f'class="grid-line" />'
            f'<text x="{pad_l}" y="{gy - 4:.1f}" class="axis-label">{gv:.1f}</text>'
        )

    for i, key in enumerate(order):
        gx = pad_l + i * group_w
        val_h = plot_h * (val_mae[key] / max_val)
        test_h = plot_h * (test_mae[key] / max_val)
        val_x = gx + group_w / 2 - gap / 2 - bar_w
        test_x = gx + group_w / 2 + gap / 2
        color = MODEL_COLOR_VAR[key]
        highlight = " bar-hero" if key == "temporal_transformer" else ""

        bars.append(
            f'<rect x="{val_x:.1f}" y="{pad_t + plot_h - val_h:.1f}" width="{bar_w:.1f}" '
            f'height="{val_h:.1f}" fill="{color}" opacity="0.55" rx="2" />'
            f'<rect x="{test_x:.1f}" y="{pad_t + plot_h - test_h:.1f}" width="{bar_w:.1f}" '
            f'height="{test_h:.1f}" fill="{color}" rx="2" class="bar{highlight}" />'
            f'<text x="{test_x + bar_w / 2:.1f}" y="{pad_t + plot_h - test_h - 6:.1f}" '
            f'class="bar-value" text-anchor="middle">{test_mae[key]:.2f}</text>'
            f'<text x="{gx + group_w / 2:.1f}" y="{H - 10}" class="axis-label" '
            f'text-anchor="middle">{esc(MODEL_LABELS[key])}</text>'
        )

    legend = (
        '<g class="legend" transform="translate(8, 2)">'
        '<rect x="0" y="0" width="10" height="10" fill="var(--c-muted)" opacity="0.55" rx="2"/>'
        '<text x="16" y="9" class="legend-label">val MAE</text>'
        '<rect x="80" y="0" width="10" height="10" fill="var(--c-muted)" rx="2"/>'
        '<text x="96" y="9" class="legend-label">test MAE (koyu)</text>'
        "</g>"
    )

    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
        f'aria-label="Model karşılaştırma çubuk grafiği">'
        f"{legend}{''.join(gridlines)}{''.join(bars)}</svg>"
    )


def feature_importance_chart(feature_attn: dict) -> str:
    items = sorted(feature_attn.items(), key=lambda kv: kv[1], reverse=True)
    max_v = items[0][1] * 1.15
    row_h = 26
    W = 640
    H = row_h * len(items) + 12
    label_w = 190
    bar_area = W - label_w - 70

    rows = []
    for i, (name, val) in enumerate(items):
        y = 8 + i * row_h
        w = bar_area * (val / max_v)
        is_top = i == 0
        color = "var(--c-gold)" if is_top else "var(--c-accent)"
        rows.append(
            f'<text x="{label_w - 10}" y="{y + row_h * 0.62:.1f}" text-anchor="end" '
            f'class="row-label">{esc(FEATURE_LABELS.get(name, name))}</text>'
            f'<rect x="{label_w}" y="{y + 3}" width="{bar_area:.1f}" height="{row_h - 10}" '
            f'class="track" rx="2" />'
            f'<rect x="{label_w}" y="{y + 3}" width="{w:.1f}" height="{row_h - 10}" '
            f'fill="{color}" rx="2" />'
            f'<text x="{label_w + bar_area + 8:.1f}" y="{y + row_h * 0.62:.1f}" class="bar-value-inline">'
            f"{val:.3f}</text>"
        )

    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart chart-tall" role="img" '
        f'aria-label="Özellik önemi (attention) yatay çubuk grafiği">{"".join(rows)}</svg>'
    )


def day_importance_chart(day_importance: dict) -> str:
    items = list(day_importance.items())
    W, H = 640, 140
    pad = 40
    plot_w = W - pad * 2
    bar_w = plot_w / len(items) * 0.5
    max_v = max(v for _, v in items) * 1.25
    min_v = min(v for _, v in items) * 0.85

    def y_of(v):
        return 16 + (H - 60) * (1 - (v - min_v) / (max_v - min_v))

    baseline_y = y_of(min_v)
    bars = []
    for i, (label, val) in enumerate(items):
        cx = pad + plot_w * (i + 0.5) / len(items)
        y = y_of(val)
        h = baseline_y - y
        bars.append(
            f'<rect x="{cx - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{max(h,1):.1f}" '
            f'class="track-strong" rx="2" />'
            f'<text x="{cx:.1f}" y="{H - 12}" class="axis-label" text-anchor="middle">{label}</text>'
        )
    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
        f'aria-label="Zaman ekseni attention grafiği">'
        f'<line x1="{pad}" y1="{baseline_y:.1f}" x2="{W-pad}" y2="{baseline_y:.1f}" class="grid-line"/>'
        f'{"".join(bars)}</svg>'
    )


def spearman_meter(value: float) -> str:
    W, H = 400, 64
    pad = 20
    track_w = W - pad * 2
    x = pad + track_w * value
    zones = [
        (0.0, 0.2, "var(--c-critical)"),
        (0.2, 0.5, "var(--c-warn)"),
        (0.5, 0.8, "var(--c-good)"),
        (0.8, 1.0, "var(--c-good)"),
    ]
    zone_rects = "".join(
        f'<rect x="{pad + track_w*a:.1f}" y="24" width="{track_w*(b-a):.1f}" height="10" '
        f'fill="{c}" opacity="0.28" />'
        for a, b, c in zones
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart chart-meter" role="img" '
        f'aria-label="Attention-permutation Spearman korelasyon göstergesi">'
        f'<rect x="{pad}" y="24" width="{track_w}" height="10" class="track" rx="5"/>'
        f"{zone_rects}"
        f'<circle cx="{x:.1f}" cy="29" r="7" fill="var(--c-gold)" stroke="var(--c-surface)" stroke-width="2"/>'
        f'<text x="{x:.1f}" y="14" text-anchor="middle" class="bar-value">{value:.2f}</text>'
        f'<text x="{pad}" y="{H-4}" class="axis-label">0.0 (ilişkisiz)</text>'
        f'<text x="{W-pad}" y="{H-4}" text-anchor="end" class="axis-label">1.0 (tam örtüşme)</text>'
        f"</svg>"
    )


def granger_chart(results: list[dict]) -> str:
    items = sorted(results, key=lambda r: r["best_p_value"])
    row_h = 30
    W = 640
    H = row_h * len(items) + 16
    label_w = 200
    bar_area = W - label_w - 78
    max_neglog = 6.0  # -log10(p) tavanı, çok küçük p'leri sıkıştırmamak için
    threshold_neglog = -math.log10(0.05)

    rows = []
    for i, r in enumerate(items):
        y = 8 + i * row_h
        neglog = min(-math.log10(max(r["best_p_value"], 1e-12)), max_neglog)
        w = bar_area * (neglog / max_neglog)
        sig = r["significant_at_0.05"]
        color = "var(--c-gold)" if sig else "var(--c-line-strong)"
        p_str = f'{r["best_p_value"]:.4f}' if r["best_p_value"] >= 0.0001 else f'{r["best_p_value"]:.1e}'
        star = " ★" if sig else ""
        rows.append(
            f'<text x="{label_w - 10}" y="{y + row_h*0.6:.1f}" text-anchor="end" class="row-label">'
            f'{esc(FEATURE_LABELS.get(r["feature"], r["feature"]))}</text>'
            f'<rect x="{label_w}" y="{y+5}" width="{bar_area:.1f}" height="{row_h-14}" class="track" rx="2"/>'
            f'<rect x="{label_w}" y="{y+5}" width="{w:.1f}" height="{row_h-14}" fill="{color}" rx="2"/>'
            f'<text x="{label_w + bar_area + 8:.1f}" y="{y + row_h*0.6:.1f}" class="bar-value-inline">'
            f"{p_str}{star}</text>"
        )

    threshold_x = label_w + bar_area * (threshold_neglog / max_neglog)
    rows.append(
        f'<line x1="{threshold_x:.1f}" y1="0" x2="{threshold_x:.1f}" y2="{H-4}" class="threshold-line"/>'
        f'<text x="{threshold_x:.1f}" y="{H-2}" class="axis-label" text-anchor="middle">p=0.05</text>'
    )

    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart chart-tall" role="img" '
        f'aria-label="Granger causality p-değerleri grafiği">{"".join(rows)}</svg>'
    )


def confounder_bars(raw_r: float, partial_r: float) -> str:
    W, H = 400, 110
    pad_l, pad_r = 190, 68
    bar_area = W - pad_l - pad_r
    max_abs = max(abs(raw_r), abs(partial_r)) * 1.2
    row_h = 40

    def bar(y, val, label, color):
        w = bar_area * (abs(val) / max_abs)
        return (
            f'<text x="{pad_l - 10}" y="{y + row_h*0.55:.1f}" text-anchor="end" class="row-label">{label}</text>'
            f'<rect x="{pad_l}" y="{y+8}" width="{bar_area:.1f}" height="{row_h-20}" class="track" rx="2"/>'
            f'<rect x="{pad_l}" y="{y+8}" width="{w:.1f}" height="{row_h-20}" fill="{color}" rx="2"/>'
            f'<text x="{pad_l + bar_area + 8:.1f}" y="{y + row_h*0.55:.1f}" class="bar-value-inline">r={val:+.3f}</text>'
        )

    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Confounder kısmi korelasyon grafiği">'
        + bar(4, raw_r, "Ham korelasyon", "var(--c-accent)")
        + bar(4 + row_h, partial_r, "Stres kontrollü", "var(--c-gold)")
        + "</svg>"
    )


def natural_experiment_bars(close_mean: float, far_mean: float, n_close: int, n_far: int) -> str:
    W, H = 300, 200
    pad_b = 30
    bar_w = 70
    gap = 60
    max_v = 10.0
    plot_h = H - pad_b - 10

    def y_of(v):
        return 10 + plot_h * (1 - v / max_v)

    x1 = W / 2 - gap / 2 - bar_w
    x2 = W / 2 + gap / 2
    y1, y2 = y_of(close_mean), y_of(far_mean)
    h1, h2 = plot_h + 10 - y1, plot_h + 10 - y2

    return (
        f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Kafein zamanlaması doğal deney grafiği">'
        f'<rect x="{x1}" y="{y1:.1f}" width="{bar_w}" height="{h1:.1f}" fill="var(--c-warn)" rx="3"/>'
        f'<rect x="{x2}" y="{y2:.1f}" width="{bar_w}" height="{h2:.1f}" fill="var(--c-good)" rx="3"/>'
        f'<text x="{x1+bar_w/2}" y="{y1-8:.1f}" text-anchor="middle" class="bar-value">{close_mean:.2f}</text>'
        f'<text x="{x2+bar_w/2}" y="{y2-8:.1f}" text-anchor="middle" class="bar-value">{far_mean:.2f}</text>'
        f'<text x="{x1+bar_w/2}" y="{H-14}" text-anchor="middle" class="axis-label">Yakın (&lt;6h)</text>'
        f'<text x="{x1+bar_w/2}" y="{H-2}" text-anchor="middle" class="axis-label">n={n_close}</text>'
        f'<text x="{x2+bar_w/2}" y="{H-14}" text-anchor="middle" class="axis-label">Uzak (≥6h)</text>'
        f'<text x="{x2+bar_w/2}" y="{H-2}" text-anchor="middle" class="axis-label">n={n_far}</text>'
        "</svg>"
    )


# ---------------------------------------------------------------------------
# Sayfa şablonu
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #EEF1F7;
  --surface: #FFFFFF;
  --surface-raised: #F7F8FC;
  --ink: #1A2032;
  --muted: #5B6178;
  --line: #D8DCE8;
  --line-strong: #C1C7DA;
  --c-accent: #4A4FB0;
  --c-accent-soft: #E4E4F7;
  --c-gold: #B8822E;
  --c-good: #2E8B6B;
  --c-warn: #A66A2E;
  --c-critical: #B8453D;
  --c-model-naive: #9AA0BF;
  --c-model-linear: #7B80D6;
  --c-model-xgb: #5B60C4;
  --c-model-tft: #B8822E;
  --shadow-card: 0 1px 2px rgba(26,32,50,0.06), 0 1px 1px rgba(26,32,50,0.04);
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0A0E1A;
    --surface: #12172A;
    --surface-raised: #171D33;
    --ink: #E8EAF6;
    --muted: #949BC4;
    --line: #262C47;
    --line-strong: #333C63;
    --c-accent: #7B80E8;
    --c-accent-soft: #23274A;
    --c-gold: #E7B75F;
    --c-good: #4FCB9E;
    --c-warn: #D99A55;
    --c-critical: #E2695F;
    --c-model-naive: #4A5078;
    --c-model-linear: #6267C4;
    --c-model-xgb: #7B80E8;
    --c-model-tft: #E7B75F;
    --shadow-card: 0 1px 2px rgba(0,0,0,0.35), 0 1px 1px rgba(0,0,0,0.25);
  }
}

:root[data-theme="dark"] {
  --bg: #0A0E1A;
  --surface: #12172A;
  --surface-raised: #171D33;
  --ink: #E8EAF6;
  --muted: #949BC4;
  --line: #262C47;
  --line-strong: #333C63;
  --c-accent: #7B80E8;
  --c-accent-soft: #23274A;
  --c-gold: #E7B75F;
  --c-good: #4FCB9E;
  --c-warn: #D99A55;
  --c-critical: #E2695F;
  --c-model-naive: #4A5078;
  --c-model-linear: #6267C4;
  --c-model-xgb: #7B80E8;
  --c-model-tft: #E7B75F;
  --shadow-card: 0 1px 2px rgba(0,0,0,0.35), 0 1px 1px rgba(0,0,0,0.25);
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  line-height: 1.5;
}

.serif {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
}

.mono, .bar-value, .bar-value-inline, .axis-label, .stat-num, table.data td.num {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-variant-numeric: tabular-nums;
}

a { color: var(--c-accent); }

.wrap { max-width: 1080px; margin: 0 auto; padding: 0 24px; }

/* ---------- top bar ---------- */
.topbar {
  border-bottom: 1px solid var(--line);
  padding: 18px 0;
}
.topbar .wrap { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.brand { display: flex; align-items: center; gap: 10px; font-family: "Iowan Old Style", Palatino, Georgia, serif; font-size: 1.15rem; letter-spacing: 0.02em; }
.brand svg { position: relative; top: -1px; }
.brand b { font-weight: 700; }
.status-pill {
  font-size: 0.78rem;
  color: var(--muted);
  border: 1px solid var(--line-strong);
  border-radius: 999px;
  padding: 4px 12px;
  white-space: nowrap;
}
.status-pill strong { color: var(--c-good); }
.nav-cta {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--c-accent);
  text-decoration: none;
  padding: 6px 4px;
  border-bottom: 1px solid transparent;
}
.nav-cta:hover { border-bottom-color: var(--c-accent); }
.nav-cta.ghost { color: var(--muted); }
.nav-cta.ghost:hover { border-bottom-color: var(--muted); }

/* ---------- splash (marka açılış anı) ---------- */
#splash {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #0B1340;
  animation: splash-fade-out 500ms ease 2100ms forwards;
}
#splash svg {
  opacity: 0;
  transform: scale(0.88);
  animation: splash-logo-in 650ms cubic-bezier(0.22, 1, 0.36, 1) 150ms forwards;
}
@keyframes splash-logo-in { to { opacity: 1; transform: scale(1); } }
@keyframes splash-fade-out { to { opacity: 0; visibility: hidden; pointer-events: none; } }
@media (prefers-reduced-motion: reduce) {
  #splash { animation: none; opacity: 0; visibility: hidden; pointer-events: none; }
  #splash svg { animation: none; opacity: 1; transform: none; }
}

/* ---------- hero (karşılama ekranı — sade, ürün odaklı) ---------- */
.hero {
  padding: 96px 0 88px;
  background:
    radial-gradient(1200px 520px at 20% -15%, var(--c-accent-soft), transparent 62%);
  border-bottom: 1px solid var(--line);
}
.hero .eyebrow {
  text-transform: uppercase;
  font-size: 0.74rem;
  letter-spacing: 0.16em;
  color: var(--c-accent);
  font-weight: 600;
  margin: 0 0 18px;
}
.hero h1 {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-weight: 500;
  font-size: clamp(2.1rem, 5vw, 3.4rem);
  line-height: 1.15;
  margin: 0 0 22px;
  max-width: 17ch;
  text-wrap: balance;
}
.hero p.lede {
  max-width: 56ch;
  color: var(--muted);
  font-size: 1.12rem;
  line-height: 1.6;
  margin: 0 0 40px;
}

.hero-actions {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}
.btn-primary {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--c-accent);
  color: #fff;
  font-size: 0.95rem;
  font-weight: 600;
  text-decoration: none;
  padding: 14px 26px;
  border-radius: 8px;
  white-space: nowrap;
}
.btn-primary:hover { opacity: 0.92; }
.btn-secondary {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: transparent;
  color: var(--ink);
  border: 1px solid var(--line-strong);
  font-size: 0.95rem;
  font-weight: 600;
  text-decoration: none;
  padding: 14px 22px;
  border-radius: 8px;
  white-space: nowrap;
}
.btn-secondary:hover { background: var(--surface-raised); }

/* ---------- araştırma raporu girişi (teknik detayların başladığı yer) ---------- */
.research-intro { padding: 44px 0; border-bottom: 1px solid var(--line); background: var(--surface-raised); }
.research-intro-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; flex-wrap: wrap; margin-bottom: 24px; }
.research-intro-lede { color: var(--muted); font-size: 0.92rem; max-width: 62ch; margin: 6px 0 0; line-height: 1.55; }

.pipeline {
  display: flex;
  gap: 0;
  overflow-x: auto;
  padding-bottom: 6px;
}
.pipeline-step {
  flex: 1 0 auto;
  min-width: 108px;
  padding: 12px 14px 12px 0;
  position: relative;
}
.pipeline-step .num {
  font-family: ui-monospace, monospace;
  font-size: 0.72rem;
  color: var(--muted);
}
.pipeline-step .name {
  font-size: 0.82rem;
  margin-top: 4px;
  color: var(--ink);
}
.pipeline-step .bar {
  margin-top: 10px;
  height: 3px;
  border-radius: 2px;
  background: var(--line-strong);
}
.pipeline-step.done .bar { background: var(--c-good); }
.pipeline-step.done .num { color: var(--c-good); }
.pipeline-step.pending .name { color: var(--muted); }

/* ---------- sections ---------- */
section { padding: 48px 0; border-bottom: 1px solid var(--line); }
section:last-of-type { border-bottom: none; }
.section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.section-head h2 {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-weight: 500;
  font-size: 1.55rem;
  margin: 0;
}
.section-head .section-note { color: var(--muted); font-size: 0.88rem; max-width: 46ch; }

.grid { display: grid; gap: 20px; }
.grid-2 { grid-template-columns: 1.15fr 0.85fr; }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
@media (max-width: 780px) {
  .grid-2, .grid-3 { grid-template-columns: 1fr; }
}

.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 22px 24px;
  box-shadow: var(--shadow-card);
}
.card h3 {
  font-size: 0.95rem;
  margin: 0 0 4px;
  font-weight: 600;
}
.card .card-sub { color: var(--muted); font-size: 0.83rem; margin: 0 0 16px; }

.stat-row { display: flex; gap: 28px; flex-wrap: wrap; margin-bottom: 18px; }
.stat { min-width: 120px; }
.stat .stat-num { font-size: 1.6rem; font-weight: 600; }
.stat .stat-num.good { color: var(--c-good); }
.stat .stat-num.gold { color: var(--c-gold); }
.stat .stat-label { font-size: 0.76rem; color: var(--muted); margin-top: 2px; }

.chart { width: 100%; height: auto; display: block; }
.chart-tall { max-height: 420px; }
.chart-meter { max-width: 420px; }

.grid-line { stroke: var(--line); stroke-width: 1; }
.threshold-line { stroke: var(--c-critical); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0.7; }
.axis-label { fill: var(--muted); font-size: 9.5px; }
.row-label { fill: var(--ink); font-size: 11px; }
.bar-value { fill: var(--ink); font-size: 11px; font-weight: 600; }
.bar-value-inline { fill: var(--muted); font-size: 10.5px; }
.legend-label { fill: var(--muted); font-size: 9.5px; }
.track { fill: var(--line); }
.track-strong { fill: var(--c-accent); opacity: 0.75; }
.bar { }
.bar-hero { filter: none; }

table.data { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
table.data th, table.data td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
table.data th { color: var(--muted); font-weight: 500; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
table.data td.num { text-align: right; }
table.data tr:last-child td { border-bottom: none; }
.table-scroll { overflow-x: auto; }

.callout {
  border: 1px solid var(--line-strong);
  border-left: 3px solid var(--c-warn);
  background: var(--surface-raised);
  border-radius: 6px;
  padding: 16px 18px;
  font-size: 0.9rem;
  color: var(--ink);
}
.callout.insight { border-left-color: var(--c-gold); }
.callout strong { color: var(--ink); }
.callout p { margin: 0 0 8px; }
.callout p:last-child { margin-bottom: 0; }

.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.tag {
  font-size: 0.72rem;
  color: var(--muted);
  border: 1px solid var(--line-strong);
  border-radius: 5px;
  padding: 3px 9px;
  font-family: ui-monospace, monospace;
}

.plan-list { margin: 0; padding-left: 1.15em; font-size: 0.9rem; }
.plan-list li { margin-bottom: 8px; }
.plan-list li::marker { color: var(--c-accent); font-family: ui-monospace, monospace; }

.limits { margin: 0; padding-left: 1.15em; font-size: 0.85rem; color: var(--muted); }
.limits li { margin-bottom: 6px; }

footer { padding: 40px 0 60px; color: var(--muted); font-size: 0.82rem; }
footer .wrap { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
"""


def build() -> str:
    baselines = _load("baseline_results.json")
    transformer = _load("transformer_results.json")
    interp = _load("interpretability_results.json")
    causality = _load("causality_results.json")

    naive_test_mae = baselines["naive"]["test"]["mae"]
    tft_test_mae = transformer["test"]["mae"]
    reduction_pct = (naive_test_mae - tft_test_mae) / naive_test_mae * 100

    xgb_test_mae = baselines["xgboost"]["test"]["mae"]
    tft_vs_xgb_pct = (xgb_test_mae - tft_test_mae) / xgb_test_mae * 100

    spearman = interp["attention_vs_permutation_spearman"]
    top_feature = max(interp["feature_importance_attention"].items(), key=lambda kv: kv[1])

    confound = causality["confounder_analysis_caffeine_stress"]
    nat_exp = causality["caffeine_timing_natural_experiment"]
    granger = causality["granger_causality"]
    n_significant = sum(1 for r in granger if r["significant_at_0.05"])

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pipeline_steps = [
        ("1", "Sentetik veri", True),
        ("2", "Ön işleme", True),
        ("3", "Baseline'lar", True),
        ("4", "Transformer", True),
        ("5", "Eğitim", True),
        ("6", "Attention", True),
        ("7", "Nedensellik", True),
        ("8", "Rapor", True),
        ("9", "Gerçek veri", True),
    ]
    n_done = sum(1 for _, _, done in pipeline_steps if done)
    pipeline_html = "".join(
        f'<div class="pipeline-step {"done" if done else "pending"}">'
        f'<div class="num">{"✓" if done else "○"} {n}</div>'
        f'<div class="name">{esc(name)}</div><div class="bar"></div></div>'
        for n, name, done in pipeline_steps
    )

    model_rows = "".join(
        f'<tr><td>{esc(MODEL_LABELS[k])}</td>'
        f'<td class="num">{baselines[k]["val"]["mae"]:.3f}</td>'
        f'<td class="num">{baselines[k]["val"]["rmse"]:.3f}</td>'
        f'<td class="num">{baselines[k]["test"]["mae"]:.3f}</td>'
        f'<td class="num">{baselines[k]["test"]["rmse"]:.3f}</td></tr>'
        for k in ["naive", "linear_regression", "xgboost"]
    ) + (
        f'<tr style="font-weight:600;"><td>{esc(MODEL_LABELS["temporal_transformer"])}</td>'
        f'<td class="num">{transformer["val"]["mae"]:.3f}</td>'
        f'<td class="num">{transformer["val"]["rmse"]:.3f}</td>'
        f'<td class="num">{transformer["test"]["mae"]:.3f}</td>'
        f'<td class="num">{transformer["test"]["rmse"]:.3f}</td></tr>'
    )

    granger_rows = "".join(
        f'<tr><td>{esc(FEATURE_LABELS.get(r["feature"], r["feature"]))}</td>'
        f'<td class="num">{r["best_p_value"]:.4f}</td>'
        f'<td class="num">{r["best_lag"]}</td>'
        f'<td>{"✓ anlamlı" if r["significant_at_0.05"] else "—"}</td></tr>'
        for r in sorted(granger, key=lambda r: r["best_p_value"])
    )

    limits_html = "".join(f"<li>{esc(l)}</li>" for l in causality["limitations"])

    html = f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SOMNIA — Sonuç Raporu</title>
<meta name="description" content="Kişisel Uyku Kalitesi için Temporal Transformer: model karşılaştırması, attention yorumlanabilirliği ve correlation-vs-causation analizi." />
<link rel="icon" type="image/svg+xml" href="favicon.svg" />
<style>{CSS}</style>
</head>
<body>

<div id="splash" aria-hidden="true">
  {logo_symbol_svg(72, mist=True)}
</div>

<header class="topbar">
  <div class="wrap">
    <div class="brand">{logo_symbol_svg(22)}<span>SOMNIA</span></div>
    <nav style="display:flex; align-items:center; gap:14px;">
      <a href="#research" class="nav-cta ghost">Araştırma raporu</a>
      <a href="{WEBAPP_URL}" class="nav-cta">Kendi verini gir →</a>
    </nav>
  </div>
</header>

<section class="hero">
  <div class="wrap">
    <p class="eyebrow">Kişisel Uyku Zekası</p>
    <h1 class="serif">"Benim uykumu gerçekten ne bozuyor?"</h1>
    <p class="lede">Kafein, stres, ekran süresi, oda sıcaklığı... Günlük alışkanlıkların uyku kaliteni nasıl etkiliyor? SOMNIA, geçmiş günlerini öğrenip bir sonraki gecen için tahmin üretir — ve daha önemlisi, hangi ilişkinin gerçek, hangisinin tesadüf olduğunu ayırt etmene yardımcı olur.</p>
    <div class="hero-actions">
      <a href="{WEBAPP_URL}" class="btn-primary">Kendi verini gir →</a>
      <a href="#research" class="btn-secondary">Araştırmayı incele ↓</a>
    </div>
  </div>
</section>

<section id="research" class="research-intro">
  <div class="wrap">
    <div class="research-intro-head">
      <div>
        <p class="eyebrow">Araştırma Raporu · {generated_at}</p>
        <p class="research-intro-lede">301 günlük davranışsal/çevresel sinyalden bir sonraki gecenin uyku kalitesini tahmin eden bir Temporal Transformer; modelin neye baktığını (attention) ve bunun gerçekten nedensel bir kanıt olup olmadığını (Granger causality, confounder analizi) ayrı ayrı sınayan tam teknik döküm aşağıda.</p>
      </div>
      <div class="status-pill"><strong>{n_done}/{len(pipeline_steps)}</strong> adım tamamlandı</div>
    </div>
    <div class="pipeline">{pipeline_html}</div>
  </div>
</section>

<section id="models">
  <div class="wrap">
    <div class="section-head">
      <h2 class="serif">Model performansı</h2>
      <p class="section-note">Walk-forward split (shuffle yok) — 205 train / 43 val / 45 test gün.</p>
    </div>
    <div class="grid grid-2">
      <div class="card">
        <h3>Naive → Linear → XGBoost → Transformer</h3>
        <p class="card-sub">Test MAE (koyu çubuk) ve val MAE (açık çubuk), sleep_quality (1–10) skalasında.</p>
        {model_comparison_chart(baselines, transformer)}
      </div>
      <div class="card">
        <div class="stat-row">
          <div class="stat">
            <div class="stat-num good">−{reduction_pct:.0f}%</div>
            <div class="stat-label">Transformer test MAE, naive'e göre</div>
          </div>
          <div class="stat">
            <div class="stat-num gold">−{tft_vs_xgb_pct:.0f}%</div>
            <div class="stat-label">Transformer test MAE, en iyi baseline'a (XGBoost) göre</div>
          </div>
        </div>
        <div class="table-scroll">
          <table class="data">
            <thead><tr><th>Model</th><th class="num">Val MAE</th><th class="num">Val RMSE</th><th class="num">Test MAE</th><th class="num">Test RMSE</th></tr></thead>
            <tbody>{model_rows}</tbody>
          </table>
        </div>
        <div class="tag-row">
          <span class="tag">d_model=32</span>
          <span class="tag">2 katman</span>
          <span class="tag">{transformer["n_parameters"]:,} parametre</span>
          <span class="tag">{transformer["epochs_trained"]} epoch (early stop)</span>
        </div>
      </div>
    </div>
  </div>
</section>

<section id="attention">
  <div class="wrap">
    <div class="section-head">
      <h2 class="serif">Attention yorumlanabilirliği</h2>
      <p class="section-note">Test seti üzerinde ölçüldü. Attention ağırlığı, modelin performansı için o özelliğin gerçekten <em>gerekli</em> olduğu anlamına gelmez — bkz. sağdaki çapraz doğrulama.</p>
    </div>
    <div class="grid grid-2">
      <div class="card">
        <h3>Özellik-ekseni attention (ortalama)</h3>
        <p class="card-sub">En çok ağırlık alan: <strong style="color:var(--c-gold);">{esc(FEATURE_LABELS.get(top_feature[0], top_feature[0]))}</strong> ({top_feature[1]:.3f})</p>
        {feature_importance_chart(interp["feature_importance_attention"])}
      </div>
      <div class="card">
        <h3>Attention vs. permutation importance</h3>
        <p class="card-sub">İki bağımsız yöntemin sıralaması ne kadar örtüşüyor? (Spearman korelasyonu)</p>
        {spearman_meter(spearman)}
        <div class="callout" style="margin-top:14px;">
          <p><strong>Orta düzeyde tutarlılık ({spearman:.2f}).</strong> Attention'ın "baktığı" özellikler ile modelin performansı için gerçekten hassas olduğu özellikler kısmen örtüşüyor, tam değil. Attention haritalarını tek başına "önem kanıtı" olarak okumamak gerekir.</p>
        </div>
        <h3 style="margin-top:22px;">Zaman ekseni: hangi geçmiş gün daha etkili?</h3>
        {day_importance_chart(interp["day_importance_attention"])}
        <p class="card-sub" style="margin-top:6px;">Dağılım oldukça düz — belirgin bir "yakın geçmiş daha önemli" örüntüsü öğrenilmemiş.</p>
      </div>
    </div>
  </div>
</section>

<section id="causality">
  <div class="wrap">
    <div class="section-head">
      <h2 class="serif">Correlation vs. Causation</h2>
      <p class="section-note">Projenin akademik merkezi: gözlenen ilişkiler ne kadar güvenilir, ve sahte (confounded) olabilir mi?</p>
    </div>

    <div class="grid grid-2" style="margin-bottom:20px;">
      <div class="card">
        <h3>Confounder analizi — kafein → uyku kalitesi</h3>
        <p class="card-sub">Stres, hem kafein tüketimini hem uyku kalitesini etkileyebilir (ground truth'ta bilerek gömülü: <code>stress_confounds_caffeine=0.35</code>).</p>
        {confounder_bars(confound["raw_pearson_r"], confound["partial_r_controlling_for_control"])}
        <p class="card-sub" style="margin-top:10px;">Stres kontrol edildiğinde ilişki zayıflıyor (p={confound["raw_p_value"]:.4f} → p={confound["partial_p_value"]:.4f}) ama tamamen kaybolmuyor — kısmi confounding var, tam değil.</p>
      </div>
      <div class="card">
        <h3>Doğal deney — kafein zamanlaması</h3>
        <p class="card-sub">Yatıştan &lt;6 saat önce kafein alınan geceler vs. daha önce alınanlar (ortalama sleep_quality).</p>
        {natural_experiment_bars(nat_exp["mean_sleep_quality_close"], nat_exp["mean_sleep_quality_far"], nat_exp["n_close_to_bed"], nat_exp["n_far_from_bed"])}
        <p class="card-sub">Fark +{nat_exp["difference"]:.2f} puan, Welch t-test p&lt;0.0001 — ama bu <strong>gözlemsel</strong> bir karşılaştırma, randomize deney değil.</p>
      </div>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <h3>Granger causality testleri</h3>
        <p class="card-sub">8 özellik × 5 lag (gün). {n_significant}/8 özellik klasik testte anlamlı (p&lt;0.05, ★ ile işaretli).</p>
        {granger_chart(granger)}
      </div>
      <div class="card">
        <h3>Metodolojik bulgu</h3>
        <div class="callout insight">
          <p><strong>Kafein zamanlaması Granger testinde anlamsız (p={next(r for r in granger if r["feature"]=="caffeine_hours_before_bed")["best_p_value"]:.3f}) ama doğal deneyde çok anlamlı (p&lt;0.0001).</strong></p>
          <p>Neden? Kafein etkisi muhtemelen <em>günler-arası gecikmeli</em> değil, <em>aynı gün eşik-tabanlı</em> (6 saat sınırı) bir etki. Granger'ın doğrusal gecikme modeli bu tür eşik etkilerini yakalamakta zayıf kalıyor.</p>
          <p>Ders: tek bir istatistiksel test yeterli değil — farklı yöntemler farklı ilişki türlerine duyarlı.</p>
        </div>
        <div class="table-scroll" style="margin-top:16px;">
          <table class="data">
            <thead><tr><th>Özellik</th><th class="num">En iyi p</th><th class="num">Lag</th><th>Sonuç</th></tr></thead>
            <tbody>{granger_rows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:20px;">
      <h3>Genel sınırlamalar</h3>
      <ul class="limits">{limits_html}</ul>
    </div>
  </div>
</section>

<section id="self-experiment">
  <div class="wrap">
    <div class="section-head">
      <h2 class="serif">Korelasyondan nedenselliğe: öz-deney</h2>
      <p class="section-note">İstatistiksel testlerin veremeyeceği kanıtı, kontrollü bir kendi-üzerinde deney verebilir.</p>
    </div>
    <div class="card">
      <ol class="plan-list">
        <li><strong>Hafta 1 (kontrol):</strong> Mevcut rutinini değiştirme; kafein saatini, stres ve ekran süresini günlüğe dürüstçe kaydet.</li>
        <li><strong>Hafta 2 (müdahale):</strong> Kafeini saat 14:00'ten sonra hiç tüketme; diğer her şeyi sabit tut.</li>
        <li><strong>Karşılaştır:</strong> İki haftanın ortalama sleep_quality'sini Welch t-test ile kıyasla — <code>src/analysis/causality.py</code> içindeki <code>caffeine_timing_natural_experiment</code> fonksiyonu bunu otomatik yapar.</li>
      </ol>
      <p class="card-sub">Tam öneri ve sınırlamalar: <a href="../reports/self_experiment_recommendation.md">reports/self_experiment_recommendation.md</a></p>
    </div>
    <div class="card" style="margin-top:20px;">
      <h3>Gerçek veriyle takip etmek için</h3>
      <p class="card-sub">Bu öz-deneyi (veya günlük rutinini) kayıt altına almak için kendi hesabınla giriş yapabileceğin bir web formu var: <a href="{WEBAPP_URL}">{WEBAPP_URL.replace("https://", "")}</a>. Girdiğin veriler aynı şemada (bu raporun üretildiği sentetik veriyle birebir) dışa aktarılır ve tüm pipeline'ı (baseline'lar, Transformer, attention, causality) üzerinde tekrar çalıştırabilirsin — kayıtların sana özeldir, başka kullanıcılar göremez.</p>
    </div>
  </div>
</section>

<footer>
  <div class="wrap">
    <div>PyTorch · pandas/numpy · scikit-learn + XGBoost · statsmodels · SOMNIA, {datetime.now().year}</div>
    <div><a href="https://github.com/ZEYDLCN/SOMNIA">github.com/ZEYDLCN/SOMNIA</a> · <a href="../docs/plan.md">proje planı</a></div>
  </div>
</footer>

<script>
  // Splash animasyonu (CSS, JS'siz de doğru çalışır) tamamlandıktan sonra
  // öğeyi DOM'dan kaldırır — erişilebilirlik ağacında gereksiz kalmasın.
  window.setTimeout(function () {{
    var el = document.getElementById('splash');
    if (el) el.remove();
  }}, 2700);
</script>

</body>
</html>
"""
    return html


def main() -> None:
    html = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Kaydedildi: {OUT_PATH}")


if __name__ == "__main__":
    main()
