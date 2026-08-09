"""
SQLite depolama katmanı — SOMNIA günlük veri girişi web formu (çok kullanıcılı).

Adım 9'un ilk parçası (bkz. docs/plan.md §3.2): her kullanıcının kendi
hesabıyla (kayıt/giriş) günlük verisini girdiği bir form. Her kullanıcının
kayıtları birbirinden İZOLE'dir (entries.user_id ile). Dışa aktarılan CSV
şeması, sentetik veriyle (data/synthetic_sleep_data.csv) BİREBİR AYNI
sütunları kullanır — böylece `src/data/preprocessing.py` ve devamındaki
tüm pipeline hiçbir değişiklik gerektirmeden bir kullanıcının gerçek
verisi üzerinde de çalışabilir (bkz. export_to_csv).

Not: Veritabanı ve dışa aktarılan CSV'ler kişisel sağlık verisi içerir;
bu yüzden .gitignore'da tutulur ve repoya işlenmez.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from datetime import date as date_cls
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "real_entries.db"
EXPORT_DIR = DATA_DIR

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

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                date TEXT NOT NULL,
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
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, date)
            )
            """
        )


# ---------------------------------------------------------------------------
# Kullanıcılar
# ---------------------------------------------------------------------------


class UsernameTakenError(Exception):
    pass


def validate_username(username: str) -> str | None:
    if not USERNAME_RE.match(username or ""):
        return "Kullanıcı adı 3-32 karakter olmalı; sadece harf, rakam, '_', '.', '-' içerebilir."
    return None


def validate_password(password: str) -> str | None:
    if not password or len(password) < 8:
        return "Şifre en az 8 karakter olmalı."
    return None


def create_user(username: str, password: str) -> int:
    """Yeni kullanıcı oluşturur, şifreyi hash'ler. Kullanıcı adı zaten
    varsa UsernameTakenError fırlatır."""
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            raise UsernameTakenError(username)
        cur = conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        return cur.lastrowid


def get_user_by_username(username: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def verify_password(user_row: sqlite3.Row, password: str) -> bool:
    return check_password_hash(user_row["password_hash"], password)


# ---------------------------------------------------------------------------
# Girdiler (kullanıcıya özel)
# ---------------------------------------------------------------------------


def upsert_entry(user_id: int, data: dict) -> None:
    """`(user_id, date)` çiftine göre INSERT OR REPLACE — aynı kullanıcı
    aynı günü tekrar doldurursa önceki kaydı düzeltir (upsert). Farklı
    kullanıcılar aynı tarihi bağımsız olarak kullanabilir."""
    payload = dict(data, user_id=user_id)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO entries (
                user_id, date, stress_score, caffeine_mg, caffeine_hours_before_bed,
                screen_time_before_bed_min, room_temp_c, noise_level_db,
                exercise_minutes, exercise_time_of_day, last_meal_delta_hr,
                sleep_duration_min, sleep_quality, updated_at
            ) VALUES (
                :user_id, :date, :stress_score, :caffeine_mg, :caffeine_hours_before_bed,
                :screen_time_before_bed_min, :room_temp_c, :noise_level_db,
                :exercise_minutes, :exercise_time_of_day, :last_meal_delta_hr,
                :sleep_duration_min, :sleep_quality, datetime('now')
            )
            ON CONFLICT(user_id, date) DO UPDATE SET
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
            payload,
        )


def delete_entry(user_id: int, entry_date: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM entries WHERE user_id = ? AND date = ?", (user_id, entry_date)
        )


def get_entry(user_id: int, entry_date: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM entries WHERE user_id = ? AND date = ?", (user_id, entry_date)
        ).fetchone()


def list_entries(user_id: int, limit: int | None = None) -> list[sqlite3.Row]:
    query = "SELECT * FROM entries WHERE user_id = ? ORDER BY date DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        return conn.execute(query, (user_id,)).fetchall()


def count_entries(user_id: int) -> int:
    with get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS n FROM entries WHERE user_id = ?", (user_id,)
        ).fetchone()["n"]


def export_path_for(username: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", username)
    return EXPORT_DIR / f"real_sleep_data_{safe}.csv"


def export_to_csv(user_id: int, username: str, out_path: Path | None = None) -> tuple[Path, int]:
    """Bir kullanıcının TÜM kayıtlarını, sentetik veriyle birebir aynı
    şemada, tarihe göre artan sırada bir CSV'ye yazar. Dosya adı
    kullanıcıya özeldir (real_sleep_data_<username>.csv) — farklı
    kullanıcıların export'ları birbirine karışmaz/üzerine yazmaz."""
    if out_path is None:
        out_path = export_path_for(username)
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM entries WHERE user_id = ? ORDER BY date ASC", (user_id,)
        ).fetchall()

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
