# Proje: Kişisel Uyku Kalitesi için Temporal Transformer

## 1. Problem Tanımı

Mevcut sleep tracker'lar (Oura, Fitbit, Apple Watch vb.) genellikle sadece **özet istatistik** verir:
> "7 saat 24 dakika uyudun, uyku skorun 82."

Ama kullanıcının asıl sorduğu soru farklı:
> "Benim uykumu **gerçekten** ne bozuyor?"

Bu proje, günlük davranışsal/çevresel sinyalleri (kafein, ekran süresi, oda sıcaklığı, gürültü, egzersiz, yemek zamanlaması, stres) bir **zaman dizisi** olarak modelleyip, bir sonraki gecenin uyku kalitesini tahmin eden ve kullanıcıya özel örüntüleri (attention üzerinden) ortaya çıkaran bir Transformer kuruyor.

## 2. Hedefler

1. Günlük özellik dizisinden (son N gün) sonraki gece uyku kalitesini tahmin eden bir **Temporal Transformer** kurmak.
2. Attention ağırlıkları üzerinden **hangi gün / hangi özelliğin** ne kadar etkili olduğunu yorumlamak.
3. **Correlation ≠ causation** ayrımını projenin merkezine koymak: model bir ilişki bulduğunda, bunun nedensel mi yoksa sadece istatistiksel bir birliktelik mi olduğunu sorgulayan bir analiz katmanı eklemek.

## 3. Veri

### Özellik seti (günlük)
| Özellik | Açıklama |
|---|---|
| `sleep_duration` | Toplam uyku süresi (dk) |
| `sleep_efficiency` | Yatakta geçen süreye göre gerçek uyku oranı |
| `room_temp` | Oda sıcaklığı (°C) |
| `noise_level` | Gece ortalama gürültü (dB, opsiyonel kategori) |
| `screen_time_before_bed` | Yatmadan önceki 1 saatteki ekran süresi (dk) |
| `caffeine_mg` + `caffeine_hours_before_bed` | Kafein miktarı ve yatıştan kaç saat önce alındığı |
| `exercise_minutes` + `exercise_time_of_day` | Egzersiz süresi ve zamanı |
| `last_meal_delta` | Son yemek ile yatış arasındaki süre |
| `stress_score` | 1–10 öz-bildirim (veya HRV proxy) |
| `sleep_quality` (target) | 1–10 öz-bildirim ya da cihaz skoru |

### Veri kaynağı stratejisi (aşamalı)
1. **Sentetik veri üreticisi** — bilinçli olarak gömülü nedensel ilişkiler içeren (örn. "kafein yatıştan 4 saat önce alınırsa uyku kalitesi düşer") bir simülatör yazacağız. Bu, modelin **gerçekten bilinen ilişkiyi bulup bulamadığını** doğrulamamızı sağlar — akademik geçerlilik testi.
2. **Gerçek kişisel veri** (opsiyonel, ileri aşama) — Apple Health / Oura / Fitbit CSV export'u + basit bir günlük (kafein, stres, ekran süresi için manuel giriş şablonu).

## 4. Model Mimarisi

**Temporal Fusion Transformer benzeri yaklaşım** — sadece "gün" bazlı değil, "gün × özellik" bazlı token'lama:

```
Girdi: [Gün t-N, ..., Gün t-1]  ×  [özellik_1, özellik_2, ..., özellik_k]
            ↓ (feature embedding + positional/temporal encoding)
    Multi-Head Self-Attention (zaman ekseninde)
            ↓
    Multi-Head Self-Attention (özellik ekseninde) — opsiyonel, TFT tarzı
            ↓
    Regression head → Sleep Quality Forecast (t)
```

- **Positional encoding**: gün indeksi + haftanın günü (döngüsel encoding, cyclical)
- **Feature embedding**: her sürekli/kategorik özellik kendi linear projeksiyonuyla d_model boyutuna taşınır
- **Çıkış**: tek adım (t+1) tahmini; ileri aşamada multi-step forecast

## 5. Yorumlanabilirlik & Correlation vs Causation Modülü

Bu proje akademik olarak burada güçleniyor. Sadece attention görselleştirmekle kalmayacağız:

1. **Attention haritaları** → hangi geçmiş gün, hangi özellik en çok ağırlık alıyor.
2. **Permutation importance / basit ablation** → attention'ı çapraz doğrulama (attention tek başına nedensellik kanıtı değildir, bunu raporda açıkça belirteceğiz).
3. **Confounder tartışması** → örn. stres hem kafein tüketimini hem uyku kalitesini etkileyebilir → sahte korelasyon riski.
4. **Basit nedensellik testi** → Granger causality veya lag'li diff-in-diff analiz; sonucu "bu korelasyondur, güçlü nedensel kanıt değildir" diye net şekilde etiketlemek.
5. **Öneri**: kullanıcıya kendi üzerinde küçük bir A/B self-deney tasarımı önerme (örn. 1 hafta kafein saatini sabitle, 1 hafta değiştir) — korelasyondan nedenselliğe geçişin doğru yolu budur.

## 6. Değerlendirme

- **Walk-forward validation** (zaman sırasına saygılı split, rastgele shuffle yok)
- **Baseline'lar**: naive (dünkü değeri tekrar et), Linear Regression, XGBoost
- Transformer'ın bu baseline'ları **gerçekten** geçip geçmediğini göstermek — küçük veri setinde Transformer'ın abartı olma riskine karşı dürüst bir kıyas
- Metrikler: MAE, RMSE

## 7. Uygulama Yol Haritası

1. [ ] Sentetik veri üreticisi (gömülü ground-truth ilişkilerle)
2. [ ] Veri ön işleme pipeline (normalizasyon, windowing, eksik veri)
3. [ ] Baseline modeller (naive, linear reg, XGBoost)
4. [ ] Temporal Transformer modeli (PyTorch, sıfırdan)
5. [ ] Eğitim döngüsü + walk-forward validation
6. [ ] Attention / feature importance görselleştirme
7. [ ] Correlation vs causation analiz modülü (ablation + Granger testi)
8. [ ] Sonuç raporu (grafikler + bulgular)
9. [ ] (Opsiyonel, sonraki aşama) Gerçek veri entegrasyonu (Apple Health/Oura parser)

## 8. Teknoloji Stack

`PyTorch` (model), `pandas` / `numpy` (veri), `scikit-learn` + `xgboost` (baseline), `matplotlib` (görselleştirme), opsiyonel `statsmodels` (Granger causality).

## 9. Riskler / Sınırlamalar

- Gerçek kişisel veri az olacaktır (günlük veri → 60-90 gün bile "az veri" sayılır); sentetik veri bu riski demo aşamasında bertaraf eder.
- Öz-bildirim (stres, uyku kalitesi) gürültülü ve öznel.
- Ölçülmeyen confounder'lar (hastalık, seyahat, jet lag) olabilir.
- Küçük veri setinde Transformer karmaşıklığı overkill olabilir — bu yüzden baseline kıyası şart.

## 10. Sıradaki Adım

Onay verirsen 1. adımdan başlıyoruz: **sentetik veri üreticisi**. Bilinçli olarak "kafein → uyku kalitesi" gibi ilişkiler gömeceğiz, sonra modelin bunu attention üzerinden bulup bulamadığını test edeceğiz.
