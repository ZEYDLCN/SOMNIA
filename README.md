# SOMNIA — Kişisel Uyku Kalitesi için Temporal Transformer

Günlük davranışsal/çevresel sinyalleri (kafein, ekran süresi, oda sıcaklığı,
gürültü, egzersiz, yemek zamanlaması, stres) bir zaman dizisi olarak
modelleyip bir sonraki gecenin uyku kalitesini tahmin eden, ve attention
üzerinden kullanıcıya özel örüntüleri ortaya çıkaran bir Temporal
Transformer projesi.

Detaylı problem tanımı, hedefler, mimari ve yol haritası için:
[`docs/plan.md`](docs/plan.md)

## Durum

- [x] 1. Sentetik veri üreticisi (gömülü ground-truth ilişkilerle) → `data/synthetic_sleep_data.csv`, `data/ground_truth.json`
- [x] 2. Veri ön işleme pipeline (normalizasyon, windowing, eksik veri) → `src/data/preprocessing.py`
- [x] 3. Baseline modeller (naive, linear reg, XGBoost) → `src/models/baselines.py`, `reports/baseline_results.json`
- [ ] 4. Temporal Transformer modeli (PyTorch, sıfırdan)
- [ ] 5. Eğitim döngüsü + walk-forward validation
- [ ] 6. Attention / feature importance görselleştirme
- [ ] 7. Correlation vs causation analiz modülü (ablation + Granger testi)
- [ ] 8. Sonuç raporu
- [ ] 9. (Opsiyonel) Gerçek veri entegrasyonu

## Proje yapısı

```
data/                       ham ve üretilmiş veri
  synthetic_sleep_data.csv  301 günlük sentetik veri
  ground_truth.json         veri üretiminde gömülü nedensel katsayılar
                             (SADECE doğrulama için — modele verilmez)
docs/
  plan.md                   proje planı
src/
  data/
    preprocessing.py        temizleme, feature engineering, windowing, split
  models/
    baselines.py            naive, linear regression, XGBoost baseline'ları
                             (sıradaki adım: Temporal Transformer)
notebooks/                  keşif / analiz defterleri
reports/                    çıktı grafikleri, sonuç raporları
  baseline_results.json     baseline MAE/RMSE (val + test)
  baseline_mae_comparison.png
```

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ön işleme pipeline'ını çalıştırma

```bash
python -m src.data.preprocessing
```

Bu, `data/synthetic_sleep_data.csv` dosyasını okur, özellik mühendisliğini
uygular (döngüsel gün encoding, kafein/stres/egzersiz türetilmiş
özellikleri), normalize eder, `window_size` günlük pencereler halinde
diziler oluşturur ve **zaman sırasına saygılı** (walk-forward,
shuffle yok) train/val/test bölmesi üretir.

## Baseline modelleri çalıştırma

```bash
python -m src.models.baselines
```

7 günlük pencereleri düzleştirip naive, linear regression ve XGBoost
modellerini eğitir; val/test üzerinde MAE/RMSE hesaplayıp
`reports/baseline_results.json` ve `reports/baseline_mae_comparison.png`
dosyalarına yazar. Bu sonuçlar, ileride kurulacak Temporal Transformer'ın
gerçekten değer katıp katmadığını göstermek için referans noktasıdır
(bkz. docs/plan.md §6).

Güncel sonuçlar (301 günlük sentetik veri, window_size=7):

| model | val MAE | val RMSE | test MAE | test RMSE |
|---|---|---|---|---|
| naive | 1.242 | 1.462 | 1.478 | 1.825 |
| linear_regression | 1.123 | 1.397 | 1.385 | 1.743 |
| xgboost | 1.030 | 1.244 | 1.039 | 1.373 |

XGBoost, hem val hem test setinde naive ve linear regression'ı geçiyor —
Transformer için aşılması gereken çıta budur.
