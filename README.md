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
- [ ] 3. Baseline modeller (naive, linear reg, XGBoost)
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
  models/                   (sıradaki adım) baseline + transformer
notebooks/                  keşif / analiz defterleri
reports/                    çıktı grafikleri, sonuç raporları
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
