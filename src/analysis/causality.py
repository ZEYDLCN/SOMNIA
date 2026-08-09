"""
Correlation vs Causation analiz modülü — SOMNIA

Adım 7 (bkz. docs/plan.md §5.3-5.5): projenin akademik merkezi. Buraya
kadar (adım 6) attention ve permutation importance ile "modelin neye
baktığını" ölçtük. Bu modül farklı bir soru soruyor: **istatistiksel
olarak nedensel bir ilişki lehine/aleyhine kanıt var mı, ve gözlenen
korelasyonlar sahte (confounded) olabilir mi?**

Üç bacak:
  1. Confounder analizi — stress hem kafein tüketimini hem uyku kalitesini
     etkileyebilir (ground_truth.json: stress_confounds_caffeine=0.35).
     Ham korelasyon ile stress_score kontrol edilmiş kısmi korelasyonu
     kıyaslıyoruz.
  2. Granger causality testi — her aday özelliğin geçmiş değerlerinin,
     sleep_quality'nin kendi geçmişinin ötesinde ek tahmin gücü sağlayıp
     sağlamadığını test eder (statsmodels).
  3. Kendi-üzerinde A/B öz-deney önerisi — korelasyondan nedenselliğe
     geçişin tek sağlam yolu budur (bkz. docs/plan.md §5.5).

ÖNEMLİ SINIRLAMA (dürüstçe belirtiyoruz): Granger causality "öngörücü
nedensellik"tir, felsefi/mekanik nedensellik DEĞİLDİR. 301 günlük, günlük
frekanslı, potansiyel olarak durağan olmayan ve ölçülmeyen confounder'lar
(hastalık, seyahat vb.) içerebilecek bu veri setinde p<0.05 bulgusu bile
"kanıt" değil, "araştırmaya değer bir sinyal"dir.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

from src.data.preprocessing import RAW_CSV, TARGET_COL, handle_missing, load_raw

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

GRANGER_CANDIDATES = [
    "stress_score",
    "caffeine_mg",
    "caffeine_hours_before_bed",
    "screen_time_before_bed_min",
    "room_temp_c",
    "noise_level_db",
    "exercise_minutes",
    "last_meal_delta_hr",
]
MAX_LAG = 5


def _load_df() -> pd.DataFrame:
    return handle_missing(load_raw(RAW_CSV))


# ---------------------------------------------------------------------------
# 1) Confounder analizi
# ---------------------------------------------------------------------------


def _residualize(y: np.ndarray, control: np.ndarray) -> np.ndarray:
    """y'yi control üzerine regresse edip artıkları (residuals) döner —
    kısmi korelasyon hesaplamak için standart yöntem."""
    X = np.column_stack([np.ones_like(control), control])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_correlation(df: pd.DataFrame, x_col: str, y_col: str, control_col: str) -> dict:
    x, y, c = df[x_col].to_numpy(), df[y_col].to_numpy(), df[control_col].to_numpy()

    raw_r, raw_p = stats.pearsonr(x, y)

    x_resid = _residualize(x, c)
    y_resid = _residualize(y, c)
    partial_r, partial_p = stats.pearsonr(x_resid, y_resid)

    return {
        "x": x_col,
        "y": y_col,
        "control": control_col,
        "raw_pearson_r": float(raw_r),
        "raw_p_value": float(raw_p),
        "partial_r_controlling_for_control": float(partial_r),
        "partial_p_value": float(partial_p),
        "attenuation": float(raw_r - partial_r),
    }


def caffeine_timing_natural_experiment(df: pd.DataFrame, ground_truth_window_hours: float = 6.0) -> dict:
    """Ground truth'ta 'kafein yatıştan 6 saatten az önce alınırsa etki
    var' varsayımını, veriyi iki gruba ayırıp basit bir fark testiyle
    (gözlemsel, RANDOMIZE EDİLMEMİŞ — bu yüzden yine korelasyondur)
    inceler."""
    close = df[df["caffeine_hours_before_bed"] < ground_truth_window_hours]
    far = df[df["caffeine_hours_before_bed"] >= ground_truth_window_hours]

    t_stat, p_val = stats.ttest_ind(
        close[TARGET_COL], far[TARGET_COL], equal_var=False
    )

    return {
        "window_hours": ground_truth_window_hours,
        "n_close_to_bed": int(len(close)),
        "n_far_from_bed": int(len(far)),
        "mean_sleep_quality_close": float(close[TARGET_COL].mean()),
        "mean_sleep_quality_far": float(far[TARGET_COL].mean()),
        "difference": float(far[TARGET_COL].mean() - close[TARGET_COL].mean()),
        "welch_t_stat": float(t_stat),
        "p_value": float(p_val),
        "note": (
            "Bu bir gözlemsel karşılaştırmadır (randomize deney değil); "
            "gruplar arası fark başka confounder'lardan (ör. o gün stres "
            "düzeyi, çalışma temposu) kaynaklanıyor olabilir."
        ),
    }


# ---------------------------------------------------------------------------
# 2) Granger causality
# ---------------------------------------------------------------------------


def granger_test_feature(df: pd.DataFrame, feature: str, target: str = TARGET_COL, maxlag: int = MAX_LAG) -> dict:
    """statsmodels sözleşmesi: data[[y, x]) ile x'in y'yi Granger-causes
    edip etmediği test edilir (x'in geçmişi, y'nin kendi geçmişinin
    ötesinde ek tahmin gücü sağlıyor mu?)."""
    data = df[[target, feature]].to_numpy(dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        except Exception as exc:  # tekil matris / yakınsama sorunları
            return {"feature": feature, "error": str(exc)}

    per_lag = {}
    for lag, res in result.items():
        p_value = res[0]["ssr_ftest"][1]
        per_lag[int(lag)] = float(p_value)

    best_lag = int(min(per_lag, key=per_lag.get))
    return {
        "feature": feature,
        "p_value_per_lag": per_lag,
        "best_lag": best_lag,
        "best_p_value": per_lag[best_lag],
        "significant_at_0.05": per_lag[best_lag] < 0.05,
    }


def run_granger_battery(df: pd.DataFrame) -> list[dict]:
    return [granger_test_feature(df, f) for f in GRANGER_CANDIDATES]


# ---------------------------------------------------------------------------
# 3) Öz-deney önerisi
# ---------------------------------------------------------------------------

SELF_EXPERIMENT_TEMPLATE = """\
# Kendi Üzerinde A/B Öz-Deney Önerisi

Model ve istatistiksel testler sana **korelasyon** gösterebilir; asıl
nedenselliği öğrenmenin tek güvenilir yolu kontrollü bir öz-deneydir.

## Önerilen tasarım (2 haftalık)

1. **Hafta 1 (kontrol)**: Mevcut rutinini değiştirme. Her gün aynı
   şekilde kafein al, aynı saatte uyu, günlüğe (stres, ekran süresi,
   kafein saati) dürüstçe not düş.
2. **Hafta 2 (müdahale)**: Tek bir değişkeni bilinçli şekilde değiştir —
   örneğin **kafeini öğleden sonra saat 14:00'ten sonra hiç tüketme**
   (bu projedeki bulgulara göre en yüksek attention + Granger sinyaline
   sahip adaylardan biri kafein zamanlaması). Diğer her şeyi (uyku
   saati, egzersiz, ekran süresi) mümkün olduğunca sabit tut.
3. Uyku kalitesini iki hafta boyunca aynı ölçekle (öz-bildirim veya
   cihaz skoru) kaydet.
4. Basit bir fark testi (iki haftanın ortalama sleep_quality farkı, ör.
   Welch t-test) uygula — `src/analysis/causality.py` içindeki
   `caffeine_timing_natural_experiment` fonksiyonu bu analizi otomatik
   yapabilir, kendi 2 haftalık verini `data/` altına ekleyip tekrar
   çalıştırabilirsin.

## Neden bu, modelden/korelasyondan daha güçlü bir kanıt?

- **Confounder kontrolü**: Kendi rutininde diğer her şeyi sabit tutarak,
  stres gibi gizli değişkenlerin etkisini büyük ölçüde ortadan
  kaldırıyorsun (ground_truth.json: `stress_confounds_caffeine=0.35` —
  bu proje sentetik veride bilerek böyle bir confounder gömdü).
- **Zamansal müdahale**: Sadece gözlemlemiyorsun, aktif olarak
  değiştiriyorsun — bu, gözlemsel Granger causality testinden çok daha
  güçlü bir nedensellik kanıtı standardıdır (randomize kontrollü deneye
  yaklaşır, n=1 olsa da).

## Sınırlamalar

- n=1 (sadece sen) — sonuç yalnızca SANA özel bir bulgu, genellenemez.
- 2 hafta kısa; mevsimsel/yaşam tarzı değişiklikleri (seyahat, hastalık)
  sonucu bozabilir — böyle bir durum olursa o günleri veri setinden
  işaretleyip çıkar.
- Placebo/beklenti etkisi mümkün (kafeini kesince "daha iyi uyumalıyım"
  beklentisiyle öz-bildirim skorların yükselebilir) — mümkünse objektif
  bir cihaz metriği (ör. uyku süresi, uyanma sayısı) de paralel takip et.
"""


def write_self_experiment_recommendation(out_dir: Path = REPORTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "self_experiment_recommendation.md"
    out_path.write_text(SELF_EXPERIMENT_TEMPLATE)
    return out_path


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------


def main() -> None:
    df = _load_df()

    print("=== 1) Confounder analizi: kafein -> uyku kalitesi, stres kontrollü ===")
    confound = partial_correlation(df, "caffeine_mg", TARGET_COL, "stress_score")
    print(
        f"Ham korelasyon (caffeine_mg, sleep_quality):        r={confound['raw_pearson_r']:+.3f}  p={confound['raw_p_value']:.4f}\n"
        f"Kısmi korelasyon (stress_score kontrollü):          r={confound['partial_r_controlling_for_control']:+.3f}  p={confound['partial_p_value']:.4f}\n"
        f"Zayıflama (attenuation): {confound['attenuation']:+.3f}"
    )
    print(
        "Yorum: Kısmi korelasyon ham korelasyondan belirgin şekilde farklıysa "
        "(zayıflıyorsa), stres'in en azından bir kısım sahte korelasyon "
        "yarattığından şüphelenilebilir. Bu YİNE DE nedensellik kanıtı değil, "
        "sadece 'confounding olası' sinyalidir."
    )

    natural_exp = caffeine_timing_natural_experiment(df)
    print(
        f"\n=== Doğal deney: kafein yatıştan <{natural_exp['window_hours']}h önce vs sonra ===\n"
        f"Yakın (n={natural_exp['n_close_to_bed']}): ort. sleep_quality={natural_exp['mean_sleep_quality_close']:.3f}\n"
        f"Uzak  (n={natural_exp['n_far_from_bed']}): ort. sleep_quality={natural_exp['mean_sleep_quality_far']:.3f}\n"
        f"Fark: {natural_exp['difference']:+.3f}  (Welch t={natural_exp['welch_t_stat']:.3f}, p={natural_exp['p_value']:.4f})"
    )

    print("\n=== 2) Granger causality testleri (maxlag=%d) ===" % MAX_LAG)
    granger_results = run_granger_battery(df)
    for res in granger_results:
        if "error" in res:
            print(f"  {res['feature']:28s} HATA: {res['error']}")
            continue
        marker = "***" if res["significant_at_0.05"] else ""
        print(
            f"  {res['feature']:28s} en iyi p={res['best_p_value']:.4f} (lag={res['best_lag']}) {marker}"
        )
    print(
        "\nUyarı: Granger causality 'öngörücü nedensellik'tir, felsefi "
        "nedensellik kanıtı değildir; kısa/durağan olmayan seride p<0.05 "
        "bile temkinli yorumlanmalı (bkz. docs/plan.md §5)."
    )

    self_exp_path = write_self_experiment_recommendation()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "causality_results.json", "w") as f:
        json.dump(
            {
                "confounder_analysis_caffeine_stress": confound,
                "caffeine_timing_natural_experiment": natural_exp,
                "granger_causality": granger_results,
                "self_experiment_recommendation_path": str(self_exp_path.relative_to(REPORTS_DIR.parent)),
                "limitations": [
                    "301 günlük tek-kişilik veri; genellenemez.",
                    "Öz-bildirim (stress_score, sleep_quality) gürültülü/öznel.",
                    "Ölçülmeyen confounder'lar (hastalık, seyahat, jet lag) olası.",
                    "Granger causality durağanlık (stationarity) varsayar; bu kontrol edilmedi.",
                    "Çoklu karşılaştırma düzeltmesi (ör. Bonferroni) uygulanmadı — "
                    "8 özellik x 5 lag test edildi, p<0.05 eşiği rastgele bazı "
                    "'anlamlı' sonuçlar üretebilir.",
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nKaydedildi: {REPORTS_DIR}/causality_results.json")
    print(f"Kaydedildi: {self_exp_path}")


if __name__ == "__main__":
    main()
