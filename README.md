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
- [x] 6. Attention / feature importance görselleştirme → `src/models/interpret.py`
- [x] 7. Correlation vs causation analiz modülü (confounder + Granger + öz-deney) → `src/analysis/causality.py`
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
    interpret.py             attention haritaları + permutation importance
                             ile çapraz doğrulama (adım 6)
  analysis/
    causality.py             confounder analizi, Granger causality testleri,
                             öz-deney (self-experiment) önerisi (adım 7)
notebooks/                  keşif / analiz defterleri
checkpoints/                 en iyi model ağırlıkları (.pt, gitignore'da)
reports/                    çıktı grafikleri, sonuç raporları
  baseline_results.json     baseline MAE/RMSE (val + test)
  baseline_mae_comparison.png
  transformer_results.json  Transformer MAE/RMSE + config
  transformer_training_curve.png
  all_model_comparison.json tüm modellerin (baseline + transformer) kıyası
  attention_feature_heatmap.png   gün × özellik attention ısı haritası
  attention_day_importance.png    hangi geçmiş gün daha etkili
  attention_vs_permutation.png    iki bağımsız yöntemin kıyası
  interpretability_results.json   sayısal sonuçlar + Spearman korelasyonu
  causality_results.json          confounder + Granger causality sonuçları
  self_experiment_recommendation.md  kullanıcıya özel A/B öz-deney önerisi
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
attention ağırlıklarını döndürüyor (`return_attn=True`) — bunlar adım
6'da görselleştirildi (aşağıda).

## Attention / feature importance görselleştirme

```bash
python -m src.models.interpret
```

Önce `train_transformer.py` çalıştırılıp `checkpoints/temporal_transformer.pt`
üretilmiş olmalı. Bu script test seti üzerinde:

1. **Özellik-ekseni attention ısı haritası** — hangi geçmiş gün, hangi
   özelliğe ne kadar "baktığı" (`reports/attention_feature_heatmap.png`).
   En yüksek ortalama attention: `stress_score`, ardından `room_temp_c`,
   `sleep_quality` (otoregresif etki) ve `exercise_minutes`.
2. **Zaman-ekseni attention** — tahmine hangi geçmiş günün ne kadar
   katkıda bulunduğu (`reports/attention_day_importance.png`). Sonuç
   oldukça düz (t-7..t-1 arası ~0.14 civarında) — model belirgin bir
   "yakın geçmiş daha önemli" örüntüsü öğrenmemiş; bu, 301 günlük veri ve
   day_persistence etkisinin (ground truth'ta 0.3) zayıf/gürültülü
   olmasıyla tutarlı bir gözlem.
3. **Permutation importance ile çapraz doğrulama** — her özelliği ayrı
   ayrı karıştırıp val MAE'deki artışı ölçer (attention'dan bağımsız bir
   yöntem). Attention-rank ile permutation-rank arasındaki Spearman
   korelasyonu **0.34** çıktı — orta düzeyde bir tutarlılık, mükemmel
   değil. Bu tam olarak plan.md §5'in uyardığı durum: **attention ağırlığı
   yüksek olması, o özelliğin modelin tahmini için gerçekten kritik
   olduğu anlamına gelmiyor.** İki yöntem kısmen örtüşüyor ama
   birbirinin garantisi değil — dolayısıyla "modelin neye baktığı" ile
   "gerçekten neyin önemli olduğu" ayrımını raporun merkezine koyuyoruz
   (bkz. `reports/attention_vs_permutation.png`,
   `reports/interpretability_results.json`).

Bu bulgular, projenin akademik iddiasını (attention ≠ nedensellik kanıtı,
hatta tek başına güvenilir bir önem ölçümü bile değil) somut veriyle
destekliyor.

## Correlation vs Causation analizi

```bash
python -m src.analysis.causality
```

Projenin akademik merkezi (bkz. docs/plan.md §5.3-5.5). Üç bacak:

**1. Confounder analizi — kafein → uyku kalitesi, stres kontrollü**

| | r | p |
|---|---|---|
| Ham korelasyon (caffeine_mg, sleep_quality) | −0.173 | 0.0026 |
| Kısmi korelasyon (stress_score kontrollü) | −0.133 | 0.0214 |

Kısmi korelasyon ham korelasyondan zayıf (attenuation −0.040) ama
tamamen kaybolmuyor — stres, kafein-uyku ilişkisinin bir kısmını
açıklıyor olabilir (ground truth'ta bilerek gömülen
`stress_confounds_caffeine=0.35` ile tutarlı) ama tek başına ilişkiyi
tam ortadan kaldırmıyor.

**Doğal deney — kafein yatıştan 6 saatten az önce vs sonra:**

Yakın (n=130): ort. sleep_quality=5.97 vs. Uzak (n=170): ort.
sleep_quality=6.61 (fark +0.64, Welch t-test p<0.0001). Bu, gözlemsel
bir karşılaştırmadır (randomize deney değil) — gruplar arası fark başka
confounder'lardan kaynaklanıyor olabilir; yine de ground truth'taki
6 saatlik pencere varsayımıyla dikkat çekici derecede tutarlı.

**2. Granger causality testleri (maxlag=5):**

| özellik | en iyi p | lag | anlamlı mı (p<0.05)? |
|---|---|---|---|
| stress_score | 0.0000 | 2 | ✅ |
| screen_time_before_bed_min | 0.109 | 1 | ❌ |
| caffeine_hours_before_bed | 0.196 | 1 | ❌ |
| last_meal_delta_hr | 0.249 | 1 | ❌ |
| noise_level_db | 0.335 | 4 | ❌ |
| caffeine_mg | 0.580 | 3 | ❌ |
| exercise_minutes | 0.709 | 1 | ❌ |
| room_temp_c | 0.834 | 3 | ❌ |

Yalnızca `stress_score` klasik Granger testinde anlamlı çıkıyor. Dikkat
çekici bir metodolojik bulgu: `caffeine_hours_before_bed` Granger testinde
anlamsız (p=0.196) ama yukarıdaki basit grup karşılaştırmasında çok
anlamlı (p<0.0001). Bunun nedeni muhtemelen kafein-uyku ilişkisinin
**günler-arası gecikmeli** değil, **aynı gün eşik-tabanlı (6 saat sınırı)**
bir etki olması — Granger'ın doğrusal gecikme modeli bu tür eşik
etkilerini yakalamakta zayıf. Bu, "tek bir istatistiksel test yeterli
değil, farklı yöntemler farklı ilişki türlerine duyarlı" dersini somut
şekilde gösteriyor.

**3. Öz-deney önerisi** — `reports/self_experiment_recommendation.md`:
2 haftalık kontrollü bir A/B tasarımı (1 hafta kafein saatini sabitle,
1 hafta belirli bir saatten sonra kafeini tamamen kes), neden bunun
korelasyondan daha güçlü bir nedensellik kanıtı olduğu, ve
sınırlamaları (n=1, placebo etkisi, kısa süre).

**Genel sınırlamalar** (`reports/causality_results.json` içinde
listelendi): 301 günlük tek-kişilik veri, öz-bildirim gürültüsü,
ölçülmeyen confounder'lar, durağanlık kontrolü yapılmadı, çoklu
karşılaştırma düzeltmesi uygulanmadı (8 özellik × 5 lag test edildi).
