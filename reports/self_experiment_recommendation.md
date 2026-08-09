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
