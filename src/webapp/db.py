"""
SQLite depolama katmanı — SOMNIA günlük veri girişi web formu.

Adım 9'un ilk parçası (bkz. docs/plan.md §3.2): kullanıcının her gün
tarayıcıdan dolduracağı bir form burada saklanır. Şema, sentetik veriyle
(data/synthetic_sleep_data.csv) BİREBİR AYNI sütunları kullanır — böylece
`src/data/preprocessing.py` ve devamındaki tüm pipeline hiçbir değişiklik
gerektirmeden gerçek veri üzerinde de çalışabilir (bkz. export_to_csv).

Not: Veritabanı ve dışa aktarılan CSV kişisel sağlık verisi içerir; bu
yüzden .gitignore'da tutulur ve repoya işlenmez.
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import date as date_cls
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "real_entries.db"
EXPORT_CSV_PATH = DATA_DIR / "real_sleep_data.csv"

CSV_COLUMNS = [
    "date",
    "day_of_week",
    "stress_score",
    "caffeine_mg",
    "caffeine_hours_before_bed",
    "screen_time_before_bed_min",
    "room_temp_c",
    "noise_level_db",
    "exercise_minutes",
    "exercise_time_of_day",
    "last_meal_delta_hr",
    "sleep_duration_min",
    "sleep_quality",
]

EXERCISE_TIME_CHOICES = ["none", "morning", "afternoon", "evening"]

FIELD_SPECS = [
    # (name, label, type, min, max, step, help)
    ("date", "Tarih", "date", None, None, None, "Bu kaydın ait olduğu gün."),
    ("sleep_duration_min", "Uyku süresi (dk)", "number", 0, 900, 1, "Bu geceki toplam uyku süresi."),
    ("sleep_quality", "Uyku kalitesi (1–10)", "number", 1, 10, 0.1, "Öz-bildirim ya da cihaz skoru."),
    ("stress_score", "Stres skoru (1–10)", "number", 1, 10, 0.1, "Bugünkü ortalama stres düzeyin."),
    ("caffeine_mg", "Kafein (mg)", "number", 0, 1000, 1, "Bugün toplam kafein alımı (yaklaşık: 1 filtre kahve ≈ 95mg)."),
    ("caffeine_hours_before_bed", "Son kafein → yatış (saat)", "number", 0, 24, 0.1, "Son kafein alımından yatışa kadar geçen süre."),
    ("screen_time_before_bed_min", "Ekran süresi, yatmadan önce (dk)", "number", 0, 300, 1, "Yatmadan önceki 1 saatteki ekran süresi."),
    ("room_temp_c", "Oda sıcaklığı (°C)", "number", 5, 35, 0.1, "Yatak odası sıcaklığı."),
    ("noise_level_db", "Gürültü (dB)", "number", 20, 90, 0.1, "Gece ortalama gürültü düzeyi (varsa)."),
    ("exercise_minutes", "Egzersiz süresi (dk)", "number", 0, 240, 1, "Bugünkü toplam egzersiz süresi."),
    ("exercise_time_of_day", "Egzersiz zamanı", "select", None, None, None, "Egzersiz yapmadıysan 'yok' seç."),
    ("last_meal_delta_hr", "Son yemek → yatış (saat)", "number", 0, 12, 0.1, "Son yemekten yatışa kadar geçen süre."),
]


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                date TEXT PRIMARY KEY,
                stress_score REAL NOT NULL,
                caffeine_mg REAL NOT NULL,
                caffeine_hours_before_bed REAL NOT NULL,
                screen_time_before_bed_min REAL NOT NULL,
                room_temp_c REAL NOT NULL,
                noise_level_db REAL NOT NULL,
                exercise_minutes REAL NOT NULL,
                exercise_time_of_day TEXT NOT NULL,
                last_meal_delta_hr REAL NOT NULL,
                sleep_duration_min REAL NOT NULL,
                sleep_quality REAL NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


def upsert_entry(data: dict) -> None:
    """`date` alanına göre INSERT OR REPLACE — aynı günü tekrar
    doldurmak, önceki kaydı düzeltir (upsert)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO entries (
                date, stress_score, caffeine_mg, caffeine_hours_before_bed,
                screen_time_before_bed_min, room_temp_c, noise_level_db,
                exercise_minutes, exercise_time_of_day, last_meal_delta_hr,
                sleep_duration_min, sleep_quality, updated_at
            ) VALUES (
                :date, :stress_score, :caffeine_mg, :caffeine_hours_before_bed,
                :screen_time_before_bed_min, :room_temp_c, :noise_level_db,
                :exercise_minutes, :exercise_time_of_day, :last_meal_delta_hr,
                :sleep_duration_min, :sleep_quality, datetime('now')
            )
            ON CONFLICT(date) DO UPDATE SET
                stress_score=excluded.stress_score,
                caffeine_mg=excluded.caffeine_mg,
                caffeine_hours_before_bed=excluded.caffeine_hours_before_bed,
                screen_time_before_bed_min=excluded.screen_time_before_bed_min,
                room_temp_c=excluded.room_temp_c,
                noise_level_db=excluded.noise_level_db,
                exercise_minutes=excluded.exercise_minutes,
                exercise_time_of_day=excluded.exercise_time_of_day,
                last_meal_delta_hr=excluded.last_meal_delta_hr,
                sleep_duration_min=excluded.sleep_duration_min,
                sleep_quality=excluded.sleep_quality,
                updated_at=datetime('now')
            """,
            data,
        )


def delete_entry(entry_date: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM entries WHERE date = ?", (entry_date,))


def get_entry(entry_date: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM entries WHERE date = ?", (entry_date,)).fetchone()


def list_entries(limit: int | None = None) -> list[sqlite3.Row]:
    query = "SELECT * FROM entries ORDER BY date DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        return conn.execute(query).fetchall()


def count_entries() -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]


def export_to_csv(out_path: Path | None = None) -> tuple[Path, int]:
    """Tüm kayıtları, sentetik veriyle birebir aynı şemada, tarihe göre
    artan sırada bir CSV'ye yazar — src/data/preprocessing.py bunu
    doğrudan (RAW_CSV yolunu değiştirerek) okuyabilir.

    `out_path` varsayılanı modül seviyesindeki EXPORT_CSV_PATH'e göre
    ÇAĞRI ANINDA çözülür (fonksiyon tanımlanırken değil) — böylece testler
    modülün EXPORT_CSV_PATH'ini monkeypatch edip gerçek kullanım kodunu
    (`db.export_to_csv()`) değişiklik yapmadan izole çalıştırabilir."""
    if out_path is None:
        out_path = EXPORT_CSV_PATH
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM entries ORDER BY date ASC").fetchall()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for r in rows:
            d = date_cls.fromisoformat(r["date"])
            writer.writerow(
                [
                    r["date"],
                    d.weekday(),  # Pazartesi=0 .. Pazar=6, sentetik veriyle aynı kural
                    r["stress_score"],
                    r["caffeine_mg"],
                    r["caffeine_hours_before_bed"],
                    r["screen_time_before_bed_min"],
                    r["room_temp_c"],
                    r["noise_level_db"],
                    r["exercise_minutes"],
                    r["exercise_time_of_day"],
                    r["last_meal_delta_hr"],
                    r["sleep_duration_min"],
                    r["sleep_quality"],
                ]
            )
    return out_path, len(rows)
