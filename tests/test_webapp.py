import importlib

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Her testte src.webapp.db'yi geçici bir SQLite dosyasına yönlendirir
    (gerçek data/ klasörünü kullanmadan izole test)."""
    from src.webapp import db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test_entries.db")
    monkeypatch.setattr(db_module, "EXPORT_CSV_PATH", tmp_path / "test_export.csv")
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


def test_index_loads_empty(client):
    c, _ = client
    resp = c.get("/")
    assert resp.status_code == 200
    assert "Henüz kayıt yok".encode() in resp.data


def test_save_entry_then_appears_in_list(client):
    c, db = client
    resp = c.post("/entries", data=VALID_FORM, follow_redirects=True)
    assert resp.status_code == 200
    assert db.count_entries() == 1
    entry = db.get_entry("2026-01-15")
    assert entry["sleep_quality"] == 7.0


def test_upsert_overwrites_same_date(client):
    c, db = client
    c.post("/entries", data=VALID_FORM, follow_redirects=True)
    updated = dict(VALID_FORM, sleep_quality="9")
    c.post("/entries", data=updated, follow_redirects=True)

    assert db.count_entries() == 1
    assert db.get_entry("2026-01-15")["sleep_quality"] == 9.0


def test_invalid_range_is_rejected(client):
    c, db = client
    bad = dict(VALID_FORM, sleep_quality="99")  # 1-10 aralığı dışında
    c.post("/entries", data=bad, follow_redirects=True)
    assert db.count_entries() == 0


def test_delete_entry(client):
    c, db = client
    c.post("/entries", data=VALID_FORM, follow_redirects=True)
    assert db.count_entries() == 1
    c.post("/entries/2026-01-15/delete", follow_redirects=True)
    assert db.count_entries() == 0


def test_export_matches_synthetic_csv_schema(client):
    import csv

    from src.data.preprocessing import RAW_CSV

    c, db = client
    c.post("/entries", data=VALID_FORM, follow_redirects=True)
    c.post("/export", follow_redirects=True)

    with open(db.EXPORT_CSV_PATH) as f:
        exported_header = next(csv.reader(f))
    with open(RAW_CSV) as f:
        synthetic_header = next(csv.reader(f))

    assert exported_header == synthetic_header
