"""
Günlük veri girişi web formu — SOMNIA (adım 9, çok kullanıcılı)

Lokal kullanım:
    python -m src.webapp.app
    -> http://127.0.0.1:5000

Her kullanıcı kendi hesabını oluşturur (/register) ve giriş yapar
(/login); her kullanıcının kayıtları birbirinden izoledir — kimse
başkasının verisini göremez/düzenleyemez (bkz. src/webapp/db.py,
entries.user_id). Giriş her ortamda (lokal dahil) zorunludur — bu,
localhost'ta bile birden fazla kişi test edebilsin diye bilinçli bir
tercih.

Deploy notu (PythonAnywhere/Render — bkz. DEPLOY.md): sadece
SOMNIA_SECRET_KEY ortam değişkeni gerekir (session imzalamak için).
Sabit bir kullanıcı adı/şifre YOK artık — herkes kendi hesabını açar.

ÖNEMLİ: Bazı ücretsiz platformlarda (ör. Render) disk KALICI DEĞİLDİR —
servis yeniden başladığında/redeploy olduğunda `data/real_entries.db`
SIFIRLANABİLİR (tüm hesaplar dahil). Düzenli aralıklarla "İndir" ile
kendi verini yedekle. Detaylar için DEPLOY.md.

Girilen veri `data/real_entries.db` (SQLite) içinde saklanır; "Dışa
Aktar" ile `data/real_sleep_data_<kullanıcı_adı>.csv`'ye — sentetik
veriyle birebir aynı şemada — export edilir, böylece
`src/data/preprocessing.py` ve devamındaki tüm pipeline (baseline'lar,
Transformer, attention, causality) bir kullanıcının gerçek verisi
üzerinde de hiçbir kod değişikliği gerektirmeden çalışabilir.
"""

from __future__ import annotations

import os
from datetime import date as date_cls

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from src.webapp import db, inference

app = Flask(__name__)
app.secret_key = os.environ.get("SOMNIA_SECRET_KEY", "somnia-local-dev-only")

# Modül import edilir edilmez (hem `python -m src.webapp.app` hem WSGI
# sunucusunun `from src.webapp.app import app` importu) tabloların var
# olduğundan emin ol. Önceden bu sadece bazı route'larda (index, save_entry,
# export) çağrılıyordu — /register veya /login ilk istek olduğunda
# (ör. taze bir veritabanında) 'no such table: users' hatası veriyordu.
db.init_db()

PUBLIC_ENDPOINTS = {"login", "register", "static"}


@app.before_request
def require_login():
    """Statik dosyalar ve login/register dışında her istek, oturum
    açmış (session['user_id']) olmayı gerektirir."""
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if session.get("user_id"):
        return None
    return redirect(url_for("login", next=request.path))


def _current_user() -> dict | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    row = db.get_user_by_id(user_id)
    return dict(row) if row else None


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        password2 = request.form.get("password2", "")

        error = db.validate_username(username) or db.validate_password(password)
        if not error and password != password2:
            error = "Şifreler eşleşmiyor."

        if not error:
            try:
                user_id = db.create_user(username, password)
            except db.UsernameTakenError:
                error = "Bu kullanıcı adı zaten alınmış."
            else:
                session["user_id"] = user_id
                session.permanent = True
                flash(f"Hoş geldin, {username}!", "success")
                return redirect(url_for("index"))

    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = db.get_user_by_username(username)
        if user and db.verify_password(user, password):
            session["user_id"] = user["id"]
            session.permanent = True
            next_url = request.form.get("next") or url_for("index")
            return redirect(next_url)
        error = "Kullanıcı adı veya şifre hatalı."

    return render_template("login.html", error=error, next=request.args.get("next", ""))


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


def _validate(form: dict) -> tuple[dict, list[str]]:
    errors: list[str] = []
    cleaned: dict = {}

    raw_date = form.get("date", "").strip()
    try:
        cleaned["date"] = date_cls.fromisoformat(raw_date).isoformat()
    except ValueError:
        errors.append("Geçerli bir tarih seç.")
        cleaned["date"] = raw_date

    for name, label, ftype, lo, hi, _step, _help in db.FIELD_SPECS:
        if name in ("date",):
            continue
        if name == "exercise_time_of_day":
            val = form.get(name, "none")
            if val not in db.EXERCISE_TIME_CHOICES:
                errors.append(f"{label}: geçersiz seçim.")
                val = "none"
            cleaned[name] = val
            continue

        raw = form.get(name, "").strip()
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label}: sayısal bir değer gir.")
            cleaned[name] = 0.0
            continue

        if lo is not None and val < lo:
            errors.append(f"{label}: {lo}'den küçük olamaz.")
        if hi is not None and val > hi:
            errors.append(f"{label}: {hi}'den büyük olamaz.")
        cleaned[name] = val

    return cleaned, errors


def _prediction_context(entries: list) -> dict:
    """Kullanıcının son kayıtlarından (en yeniden en eskiye sıralı)
    model tahmini üretmeyi dener. Model dosyası eksikse/çıkarım
    başarısız olursa sayfa ÇÖKMEZ — sadece tahmin kartı gösterilmez."""
    try:
        w = inference.window_size()
    except Exception:
        return {"available": False}

    if len(entries) < w:
        return {"available": False, "have": len(entries), "need": w}

    try:
        pred = inference.predict_next_sleep_quality(entries[:w])
    except Exception:
        return {"available": False, "have": len(entries), "need": w}

    return {"available": True, "value": pred, "window": w}


@app.route("/", methods=["GET"])
def index():
    db.init_db()
    user = _current_user()

    edit_date = request.args.get("edit")
    prefill = None
    if edit_date:
        row = db.get_entry(user["id"], edit_date)
        if row:
            prefill = dict(row)

    entries = db.list_entries(user["id"], limit=30)
    fields_by_name = {
        name: {"label": label, "type": ftype, "min": lo, "max": hi, "step": step, "help": help_}
        for name, label, ftype, lo, hi, step, help_ in db.FIELD_SPECS
    }
    return render_template(
        "index.html",
        fields=fields_by_name,
        exercise_choices=db.EXERCISE_TIME_CHOICES,
        entries=entries,
        prefill=prefill,
        today=date_cls.today().isoformat(),
        n_entries=db.count_entries(user["id"]),
        username=user["username"],
        prediction=_prediction_context(entries),
    )


@app.route("/entries", methods=["POST"])
def save_entry():
    db.init_db()
    user = _current_user()
    cleaned, errors = _validate(request.form)
    if errors:
        for e in errors:
            flash(e, "error")
        return redirect(url_for("index", edit=cleaned.get("date") or None))

    db.upsert_entry(user["id"], cleaned)
    flash(f"{cleaned['date']} için kayıt kaydedildi.", "success")
    return redirect(url_for("index"))


@app.route("/entries/<entry_date>/delete", methods=["POST"])
def delete_entry(entry_date: str):
    user = _current_user()
    db.delete_entry(user["id"], entry_date)
    flash(f"{entry_date} kaydı silindi.", "success")
    return redirect(url_for("index"))


@app.route("/export", methods=["POST"])
def export():
    db.init_db()
    user = _current_user()
    if db.count_entries(user["id"]) == 0:
        flash("Dışa aktarılacak kayıt yok.", "error")
        return redirect(url_for("index"))
    path, n = db.export_to_csv(user["id"], user["username"])
    flash(f"{n} kayıt {path.name} dosyasına aktarıldı.", "success")
    return redirect(url_for("index"))


@app.route("/export/download", methods=["GET"])
def download_export():
    user = _current_user()
    path = db.export_path_for(user["username"])
    if not path.exists():
        flash("Önce 'Dışa Aktar'a bas.", "error")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name=path.name)


def main() -> None:
    db.init_db()
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
