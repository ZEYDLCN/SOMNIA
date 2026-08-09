"""
Günlük veri girişi web formu — SOMNIA (adım 9)

Kullanım:
    python -m src.webapp.app
    -> http://127.0.0.1:5000

Bu, kişisel/lokal kullanım için tasarlanmış küçük bir Flask uygulamasıdır
(kimlik doğrulama YOK — sadece localhost'ta veya güvenilir bir ağda
çalıştırılmalı, internete açık şekilde deploy edilmemeli). Girilen veri
`data/real_entries.db` (SQLite) içinde saklanır; "Dışa Aktar" ile
`data/real_sleep_data.csv`'ye — sentetik veriyle birebir aynı şemada —
export edilir, böylece `src/data/preprocessing.py` ve devamındaki tüm
pipeline (baseline'lar, Transformer, attention, causality) gerçek veri
üzerinde de hiçbir kod değişikliği gerektirmeden çalışabilir.
"""

from __future__ import annotations

from datetime import date as date_cls

from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from src.webapp import db

app = Flask(__name__)
app.secret_key = "somnia-local-dev-only"  # yalnızca flash mesajları için; kimlik doğrulama değil


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


@app.route("/", methods=["GET"])
def index():
    db.init_db()

    edit_date = request.args.get("edit")
    prefill = None
    if edit_date:
        row = db.get_entry(edit_date)
        if row:
            prefill = dict(row)

    entries = db.list_entries(limit=30)
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
        n_entries=db.count_entries(),
    )


@app.route("/entries", methods=["POST"])
def save_entry():
    db.init_db()
    cleaned, errors = _validate(request.form)
    if errors:
        for e in errors:
            flash(e, "error")
        return redirect(url_for("index", edit=cleaned.get("date") or None))

    db.upsert_entry(cleaned)
    flash(f"{cleaned['date']} için kayıt kaydedildi.", "success")
    return redirect(url_for("index"))


@app.route("/entries/<entry_date>/delete", methods=["POST"])
def delete_entry(entry_date: str):
    db.delete_entry(entry_date)
    flash(f"{entry_date} kaydı silindi.", "success")
    return redirect(url_for("index"))


@app.route("/export", methods=["POST"])
def export():
    db.init_db()
    if db.count_entries() == 0:
        flash("Dışa aktarılacak kayıt yok.", "error")
        return redirect(url_for("index"))
    path, n = db.export_to_csv()
    flash(f"{n} kayıt {path.name} dosyasına aktarıldı.", "success")
    return redirect(url_for("index"))


@app.route("/export/download", methods=["GET"])
def download_export():
    if not db.EXPORT_CSV_PATH.exists():
        flash("Önce 'Dışa Aktar'a bas.", "error")
        return redirect(url_for("index"))
    return send_file(db.EXPORT_CSV_PATH, as_attachment=True, download_name="real_sleep_data.csv")


def main() -> None:
    db.init_db()
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
