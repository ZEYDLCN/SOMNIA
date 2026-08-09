# Deployment — SOMNIA

Proje iki ayrı, birbirinden bağımsız parça olarak deploy edilir; ikisi de
**ücretsiz**:

| Parça | Nedir | Nereye | Neden ayrı |
|---|---|---|---|
| `docs/index.html` | statik sonuç raporu (dashboard) | **GitHub Pages** | JS/backend gerektirmez, sınırsız süre ücretsiz |
| `src/webapp/` | günlük veri girişi formu (Flask + SQLite) | **PythonAnywhere free tier** (alternatif: Render) | Sunucu tarafı state gerektirir, GitHub Pages statik olduğu için çalıştıramaz |

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

## 2. Veri girişi formu → PythonAnywhere (free tier, önerilen)

Render'ın free tier'ı build kuyruğunda uzun süre bekletebiliyor ve
sürekli "upgrade" hatırlatması gösteriyor. **PythonAnywhere** gerçekten
ücretsiz (kredi kartı istemiyor) ve **soğuk başlangıç/uykuya geçme
sorunu yok** — web app'in her zaman ayakta. Tek fark: Render gibi
git push'ta otomatik deploy yok, kod değişince elle `git pull` +
"Reload" gerekiyor (aşağıda anlatılıyor).

### Adımlar

1. [pythonanywhere.com](https://www.pythonanywhere.com) → **Pricing & signup** → **Create a Beginner account** (ücretsiz, kart istemiyor).
2. Dashboard → **Consoles** → **Bash** ile yeni bir konsol aç.
3. Repoyu klonla ve webapp bağımlılıklarını kur:
   ```bash
   git clone https://github.com/ZEYDLCN/SOMNIA.git
   cd SOMNIA
   pip install --user -r requirements-webapp.txt
   ```
4. Dashboard → **Web** sekmesi → **Add a new web app** → domain'i onayla
   (`<kullanıcı-adın>.pythonanywhere.com`) → **Manual configuration**
   (Flask şablonunu değil, "Manual configuration" seç) → **Python 3.10**
   (veya en güncel seçenek).
5. Web sekmesinde **Code** bölümünde:
   - **Source code / Working directory**: `/home/<kullanıcı-adın>/SOMNIA`
   - **WSGI configuration file** linkine tıkla, dosyanın TAMAMINI sil ve
     şunu yaz (kendi PythonAnywhere kullanıcı adınla):
     ```python
     import sys, os

     path = '/home/<kullanıcı-adın>/SOMNIA'
     if path not in sys.path:
         sys.path.append(path)

     os.environ['SOMNIA_SECRET_KEY'] = 'rastgele-uzun-bir-metin'

     from src.webapp.app import app as application
     ```
     (`SOMNIA_SECRET_KEY` sadece oturum çerezini imzalamak için —
     rastgele, uzun, kimseyle paylaşmadığın bir metin olsun. Uygulama
     artık çok kullanıcılı: sabit bir kullanıcı adı/şifre YOK, herkes
     kendi hesabını `/register`'dan açıyor.)
6. **PythonAnywhere → Web → Security → Password protection**'ın
   **kapalı** olduğundan emin ol (varsa kaldır) — bu, platformun kendi
   ayrı koruması, uygulamanın kendi giriş sistemiyle çakışır.
7. Aynı **Web** sekmesinde yeşil **Reload** butonuna bas.
8. `https://<kullanıcı-adın>.pythonanywhere.com/register` adresine
   giderek ilk hesabını oluştur. Bundan sonra herkes kendi hesabını
   `/register`'dan açıp `/login`'den giriş yapabilir — her kullanıcının
   kayıtları birbirinden bağımsızdır (kimse başkasının verisini göremez).

### Kod güncellediğinde (redeploy)

Otomatik değil — her `docs/index.html` veya `src/webapp/` değişikliğinden
sonra:
```bash
cd ~/SOMNIA && git pull
```
sonra **Web** sekmesinden **Reload**'a bas. 2 adım, ~10 saniye.

### Sınırlamalar (free tier)

- Günlük CPU saniyesi kotası var (~100 sn/gün) — kişisel/günlük tek
  kullanıcı trafiği için fazlasıyla yeterli.
- Disk kalıcıdır (Render'ın aksine) — `data/real_entries.db` silinmiyor,
  yine de düzenli "İndir" ile CSV yedeği almak iyi bir alışkanlık.
- Özel domain yok (free tier'da `pythonanywhere.com` alt alan adı
  zorunlu) — kişisel kullanım için sorun değil.

### Neden ayrı `requirements-webapp.txt`?

Ana `requirements.txt` (torch, xgboost, statsmodels...) sadece ML
pipeline'ı içindir; webapp'in bunlardan HİÇBİRİNE ihtiyacı yok (sadece
Flask + stdlib `sqlite3`). Bu ayrım, hem PythonAnywhere'in disk/CPU
kotasına hem Render'ın build sınırlarına takılmamak için var.

---

## 2b. Alternatif: Render.com (free tier)

Render'ı tercih edersen `render.yaml` hâlâ repoda ve hazır. Bilmen
gerekenler:

- **Disk kalıcı değildir** — servis yeniden başladığında
  (redeploy/çökme/~15 dk hareketsizlik sonrası uyanma)
  `data/real_entries.db` sıfırlanabilir; düzenli "İndir" ile yedek al.
- **Soğuk başlangıç** — ~15 dk istek gelmezse uykuya geçer, sıradaki
  istek 30-60 sn gecikebilir.
- **Kurulum**: [render.com](https://render.com) → **New + → Blueprint**
  → `ZEYDLCN/SOMNIA` seç → Render `render.yaml`'ı otomatik bulur
  (`SOMNIA_SECRET_KEY` otomatik üretilir) → **Apply**. Uygulama çok
  kullanıcılı — ilk açılışta `/register`'dan bir hesap oluştur.
- Blueprint yerine tek tek kurmak istersen: **New + → Web Service** →
  Build: `pip install -r requirements-webapp.txt` → Start:
  `gunicorn -w 2 -b 0.0.0.0:$PORT src.webapp.app:app` → Plan: Free.
