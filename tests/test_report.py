from src.report.build_report import build


def test_report_builds_and_contains_key_numbers():
    html = build()

    assert "<!doctype html>" in html.lower()
    assert "SOMNIA" in html
    # Model karşılaştırma tablosundaki değerler metinde geçmeli.
    assert "Temporal Transformer" in html
    assert "XGBoost" in html
    # Attention ve causality bulgularının anahtar kelimeleri sayfada olmalı.
    assert "Granger" in html
    assert "Spearman" in html
    assert "confound" in html.lower() or "kısmi korelasyon" in html.lower()

    # Tema token'ları hem açık hem koyu blokta tanımlı olmalı (theme-aware).
    assert "prefers-color-scheme: dark" in html
    assert ':root[data-theme="dark"]' in html
