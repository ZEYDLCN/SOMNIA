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
- [x] 4. Temporal Transformer modeli (PyTorch, sıfırdan) → `src/models/transformer.py`
- [x] 5. Eğitim döngüsü + walk-forward validation → `src/models/train_transformer.py`
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
    transformer.py           Temporal Transformer mimarisi (özellik-ekseni +
                             zaman-ekseni self-attention, sıfırdan PyTorch)
    train_transformer.py     eğitim döngüsü, walk-forward val ile early
                             stopping, baseline karşılaştırması
notebooks/                  keşif / analiz defterleri
checkpoints/                 en iyi model ağırlıkları (.pt, gitignore'da)
reports/                    çıktı grafikleri, sonuç raporları
  baseline_results.json     baseline MAE/RMSE (val + test)
  baseline_mae_comparison.png
  transformer_results.json  Transformer MAE/RMSE + config
  transformer_training_curve.png
  all_model_comparison.json tüm modellerin (baseline + transformer) kıyası
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

## Temporal Transformer eğitimi

```bash
python -m src.models.train_transformer
```

Mimari (bkz. docs/plan.md §4): her (gün, özellik) hücresi kendi linear
projeksiyonuyla `d_model` boyutuna taşınır → özellik ekseninde self-attention
ile gün başına tek bir token'a havuzlanır → sinüsoidal zaman encoding
eklenir → çok katmanlı zaman-ekseni self-attention → öğrenilen query ile
attention-pooling → regresyon başlığı. Küçük veri seti riskine karşı
(`docs/plan.md §9`) model kasıtlı küçük tutuldu (`d_model=32`, 2 katman,
~27K parametre, dropout=0.2) ve val MAE'ye göre early stopping uygulanır.

Güncel sonuçlar:

| model | val MAE | val RMSE | test MAE | test RMSE |
|---|---|---|---|---|
| naive | 1.242 | 1.462 | 1.478 | 1.825 |
| linear_regression | 1.123 | 1.397 | 1.385 | 1.743 |
| xgboost | 1.030 | 1.244 | 1.039 | 1.373 |
| **temporal_transformer** | **0.901** | **1.140** | **0.975** | **1.322** |

Transformer, hem val hem test setinde tüm baseline'ları geçiyor — 301
günlük küçük bir veri setinde bile dikkat mekanizmasının basit
modellerin yakalayamadığı örüntüleri (gecikmeli/etkileşimli etkiler)
öğrenebildiğine işaret ediyor. Model ayrıca özellik-ekseni ve zaman-ekseni
attention ağırlıklarını döndürüyor (`return_attn=True`) — bunlar sıradaki
adımda (6) görselleştirilecek.
