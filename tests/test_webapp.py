import importlib

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Her testte src.webapp.db'yi geçici bir SQLite dosyasına yönlendirir
    (gerçek data/ klasörünü kullanmadan izole test)."""
    from src.webapp import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test_entries.db")
    monkeypatch.setattr(db_module, "EXPORT_DIR", tmp_path)
    db_module.init_db()

    from src.webapp import app as app_module

    importlib.reload(app_module)  # db referanslarının yamalı modülü kullanmasını garanti eder
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c, db_module


VALID_FORM = {
    "date": "2026-01-15",
    "sleep_duration_min": "430",
    "sleep_quality": "7",
    "stress_score": "4",
    "caffeine_mg": "120",
    "caffeine_hours_before_bed": "8",
    "screen_time_before_bed_min": "30",
    "exercise_minutes": "0",
    "exercise_time_of_day": "none",
    "last_meal_delta_hr": "2.5",
    "room_temp_c": "20",
    "noise_level_db": "30",
}


def _register(c, username="alice", password="password123"):
    return c.post(
        "/register",
        data={"username": username, "password": password, "password2": password},
        follow_redirects=True,
    )


def _login(c, username="alice", password="password123"):
    return c.post("/login", data={"username": username, "password": password}, follow_redirects=True)


def test_register_works_on_fresh_db_without_prior_init(tmp_path, monkeypatch):
    """Regresyon: PythonAnywhere'de görülen 'no such table: users' hatası.
    Önceden db.init_db() sadece bazı route'larda (index, save_entry,
    export) çağrılıyordu; /register veritabanı dosyası hiç yokken ilk
    istek olursa tablo bulunamıyordu. app.py artık modül import
    edilirken init_db() çağırıyor — bu test o garantiyi, fixture'ın
    kendi (maskeleyen) init_db() çağrısı OLMADAN doğrular."""
    from src.webapp import db as db_module

    db_path = tmp_path / "fresh.db"
    assert not db_path.exists()
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    monkeypatch.setattr(db_module, "EXPORT_DIR", tmp_path)
    # BİLEREK db_module.init_db() çağrılmıyor — app modülünün kendi
    # başına tabloyu oluşturması gerekiyor.

    import importlib as _importlib

    from src.webapp import app as app_module

    _importlib.reload(app_module)  # reload sırasında modül seviyesinde db.init_db() çalışır
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        resp = _register(c, "freshuser", "password123")
        assert resp.status_code == 200
        assert "Henüz kayıt yok".encode() in resp.data
        assert db_module.get_user_by_username("freshuser") is not None


# ---------------------------------------------------------------------------
# Kimlik doğrulama gerekliliği
# ---------------------------------------------------------------------------


def test_index_requires_login(client):
    c, _ = client
    resp = c.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_register_then_auto_logged_in(client):
    c, db = client
    resp = _register(c)
    assert resp.status_code == 200
    assert "Henüz kayıt yok".encode() in resp.data
    assert db.get_user_by_username("alice") is not None


def test_register_duplicate_username_rejected(client):
    c, _ = client
    _register(c, "alice")
    c.post("/logout")
    resp = _register(c, "alice")
    assert "zaten alınmış".encode() in resp.data


def test_register_password_mismatch_rejected(client):
    c, _ = client
    resp = c.post(
        "/register",
        data={"username": "bob", "password": "password123", "password2": "different"},
        follow_redirects=True,
    )
    assert "eşleşmiyor".encode() in resp.data


def test_register_short_password_rejected(client):
    c, db = client
    c.post(
        "/register",
        data={"username": "bob", "password": "short", "password2": "short"},
        follow_redirects=True,
    )
    assert db.get_user_by_username("bob") is None


def test_login_wrong_password_rejected(client):
    c, _ = client
    _register(c, "alice")
    c.post("/logout")
    resp = _login(c, "alice", "wrongpassword")
    assert "hatalı".encode() in resp.data


def test_logout_revokes_access(client):
    c, _ = client
    _register(c, "alice")
    assert c.get("/").status_code == 200
    c.post("/logout", follow_redirects=True)
    resp = c.get("/", follow_redirects=False)
    assert resp.status_code == 302


# ---------------------------------------------------------------------------
# Kayıt CRUD + kullanıcı izolasyonu
# ---------------------------------------------------------------------------


def test_save_entry_then_appears_in_list(client):
    c, db = client
    _register(c, "alice")
    c.post("/entries", data=VALID_FORM, follow_redirects=True)

    user = db.get_user_by_username("alice")
    assert db.count_entries(user["id"]) == 1
    entry = db.get_entry(user["id"], "2026-01-15")
    assert entry["sleep_quality"] == 7.0


def test_upsert_overwrites_same_date(client):
    c, db = client
    _register(c, "alice")
    c.post("/entries", data=VALID_FORM, follow_redirects=True)
    updated = dict(VALID_FORM, sleep_quality="9")
    c.post("/entries", data=updated, follow_redirects=True)

    user = db.get_user_by_username("alice")
    assert db.count_entries(user["id"]) == 1
    assert db.get_entry(user["id"], "2026-01-15")["sleep_quality"] == 9.0


def test_invalid_range_is_rejected(client):
    c, db = client
    _register(c, "alice")
    bad = dict(VALID_FORM, sleep_quality="99")  # 1-10 aralığı dışında
    c.post("/entries", data=bad, follow_redirects=True)

    user = db.get_user_by_username("alice")
    assert db.count_entries(user["id"]) == 0


def test_delete_entry(client):
    c, db = client
    _register(c, "alice")
    c.post("/entries", data=VALID_FORM, follow_redirects=True)
    user = db.get_user_by_username("alice")
    assert db.count_entries(user["id"]) == 1

    c.post("/entries/2026-01-15/delete", follow_redirects=True)
    assert db.count_entries(user["id"]) == 0


def test_two_users_have_isolated_entries(client):
    c, db = client
    _register(c, "alice")
    c.post("/entries", data=VALID_FORM, follow_redirects=True)
    c.post("/logout")

    _register(c, "bob")
    resp = c.get("/")
    # bob henüz kayıt girmedi; alice'in kaydını GÖRMEMELİ.
    assert "Henüz kayıt yok".encode() in resp.data

    alice = db.get_user_by_username("alice")
    bob = db.get_user_by_username("bob")
    assert db.count_entries(alice["id"]) == 1
    assert db.count_entries(bob["id"]) == 0


def test_user_cannot_delete_another_users_entry(client):
    c, db = client
    _register(c, "alice")
    c.post("/entries", data=VALID_FORM, follow_redirects=True)
    c.post("/logout")

    _register(c, "bob")
    # bob, alice'in tarihini silmeye çalışıyor — kendi user_id'siyle
    # sorgulandığı için hiçbir şey silinmemeli.
    c.post("/entries/2026-01-15/delete", follow_redirects=True)

    alice = db.get_user_by_username("alice")
    assert db.count_entries(alice["id"]) == 1


def test_export_matches_synthetic_csv_schema(client):
    import csv

    from src.data.preprocessing import RAW_CSV

    c, db = client
    _register(c, "alice")
    c.post("/entries", data=VALID_FORM, follow_redirects=True)
    c.post("/export", follow_redirects=True)

    user = db.get_user_by_username("alice")
    export_path = db.export_path_for(user["username"])
    with open(export_path) as f:
        exported_header = next(csv.reader(f))
    with open(RAW_CSV) as f:
        synthetic_header = next(csv.reader(f))

    assert exported_header == synthetic_header


# ---------------------------------------------------------------------------
# "Bu geceki tahmin" (ONNX model çıkarımı)
# ---------------------------------------------------------------------------


def test_prediction_card_shows_progress_before_enough_data(client):
    c, _ = client
    _register(c, "alice")
    resp = c.post("/entries", data=VALID_FORM, follow_redirects=True)
    assert "0/7 gün".encode() not in resp.data  # 1 kayıt girildi
    assert "gün daha gir".encode() in resp.data


def test_prediction_appears_after_seven_days(client):
    import datetime as dt

    from src.webapp.inference import ONNX_PATH

    if not ONNX_PATH.exists():
        import pytest as _pytest

        _pytest.skip("Model export edilmemiş (src/models/export_onnx.py çalıştırılmalı).")

    c, _ = client
    _register(c, "alice")

    base = dt.date(2026, 1, 1)
    resp = None
    for i in range(7):
        entry = dict(VALID_FORM, date=(base + dt.timedelta(days=i)).isoformat())
        resp = c.post("/entries", data=entry, follow_redirects=True)

    assert "Bu geceki tahmin".encode() in resp.data
    assert "/10".encode() in resp.data
