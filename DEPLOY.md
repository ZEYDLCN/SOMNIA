# Deployment — SOMNIA

Proje iki ayrı, birbirinden bağımsız parça olarak deploy edilir; ikisi de
**ücretsiz**:

| Parça | Nedir | Nereye | Neden ayrı |
|---|---|---|---|
| `docs/index.html` | statik sonuç raporu (dashboard) | **GitHub Pages** | JS/backend gerektirmez, sınırsız süre ücretsiz |
| `src/webapp/` | günlük veri girişi formu (Flask + SQLite) | **Render.com free tier** | Sunucu tarafı state gerektirir, GitHub Pages statik olduğu için çalıştıramaz |

---

## 1. Sonuç raporu → GitHub Pages

Gerekli dosya (`.github/workflows/deploy-pages.yml`) zaten repoda; her
`main`'e push'ta `docs/` klasörünü otomatik yayınlar. Tek yapman gereken,
**bir kerelik** bir ayarı açmak (API üzerinden yapılamıyor, GitHub bunu
repo admin'ine bırakıyor):

1. GitHub'da repo → **Settings → Pages**
2. **Source** → **GitHub Actions** seç (branch/klasör seçmene gerek yok,
   workflow zaten hazır)
3. Settings → **Actions → General** altında Actions'ın etkin olduğundan
   emin ol (varsayılan olarak açıktır)
4. Kaydet — birkaç dakika içinde `https://zeydlcn.github.io/SOMNIA/`
   adresinde canlı olur

Her `docs/index.html` güncellemesinde (`python -m src.report.build_report`
+ push) site otomatik yeniden yayınlanır.

---

## 2. Veri girişi formu → Render.com (free tier)

Bu form kişisel sağlık verisi topladığı için **HTTP Basic Auth ile
korunmalı** — `render.yaml` bunu zaten `SOMNIA_USERNAME` /
`SOMNIA_PASSWORD` ortam değişkenleriyle destekliyor.

### ⚠️ Önce oku: free tier'ın iki sınırlaması

1. **Disk kalıcı değildir.** Render'ın ücretsiz web servisi planında
   diskler ephemeral'dır — servis yeniden başladığında (redeploy, çökme,
   ~15 dk hareketsizlik sonrası uyku/uyanma) `data/real_entries.db`
   **sıfırlanabilir**. Bu bir demo/kişisel-kullanım riski olarak kabul
   edildi (bkz. konuşma geçmişi) — **düzenli aralıklarla "İndir"
   butonuyla CSV yedeği al.**
2. **Soğuk başlangıç.** ~15 dakika istek gelmezse servis uykuya geçer;
   sıradaki istek 30-60 saniye gecikebilir. Kişisel/günlük kullanım için
   sorun değil.

Kabul ediyorsan devam:

### Adımlar

1. [render.com](https://render.com) üzerinde ücretsiz hesap aç (GitHub ile giriş yapılabilir).
2. **New +** → **Blueprint** → `ZEYDLCN/SOMNIA` reposunu seç.
3. Render, repodaki `render.yaml`'ı otomatik bulur ve `somnia-webapp`
   servisini önerir (plan: free, build: `requirements-webapp.txt` —
   torch/xgboost YOK, sadece Flask+gunicorn, hızlı build).
4. **Apply** etmeden önce env değişkenlerini gir:
   - `SOMNIA_USERNAME`: kendi kullanıcı adın
   - `SOMNIA_PASSWORD`: güçlü bir şifre
   - `SOMNIA_SECRET_KEY`: boş bırak, Render otomatik üretir
5. **Apply** — birkaç dakika içinde `https://somnia-webapp.onrender.com`
   (veya Render'ın verdiği adres) üzerinde, tarayıcı Basic Auth
   penceresi soracak şekilde canlı olur.

### Manuel kurulum (Blueprint kullanmadan)

Blueprint yerine tek tek de kurabilirsin: **New + → Web Service** → repo
seç → Build Command: `pip install -r requirements-webapp.txt` → Start
Command: `gunicorn -w 2 -b 0.0.0.0:$PORT src.webapp.app:app` → Plan:
Free → aynı üç env değişkenini gir.

### Neden ayrı `requirements-webapp.txt`?

Ana `requirements.txt` (torch, xgboost, statsmodels...) sadece ML
pipeline'ı içindir; webapp'in bunlardan HİÇBİRİNE ihtiyacı yok (sadece
Flask + stdlib `sqlite3`). Onları da kurmaya çalışsaydık, free tier'ın
build süresi/disk sınırlarına takılma riski olurdu — bu yüzden ayrı,
hafif bir dosya kullanıyoruz.
