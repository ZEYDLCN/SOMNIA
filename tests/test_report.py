from src.report.build_report import _compute, build_landing, build_report_page


def test_landing_page_is_marketing_only_no_technical_jargon():
    """Kullanıcı isteği: ilk açılan sayfa (docs/index.html) saf bir ürün
    karşılama ekranı olmalı — teknik rapor içeriği (model tabloları,
    Granger/Spearman gibi terimler) burada GÖRÜNMEMELİ, ayrı sayfada
    (report.html) olmalı."""
    data = _compute()
    html = build_landing(data)

    assert "<!doctype html>" in html.lower()
    assert "SOMNIA" in html
    assert "Kendi verini gir" in html
    assert "id=\"splash\"" in html  # marka açılış animasyonu

    # Teknik rapor içeriği landing'de OLMAMALI.
    for technical_term in ("Granger", "Spearman", "XGBoost", "Walk-forward"):
        assert technical_term not in html

    # Tema token'ları hem açık hem koyu blokta tanımlı olmalı (theme-aware).
    assert "prefers-color-scheme: dark" in html
    assert ':root[data-theme="dark"]' in html


def test_report_page_contains_key_numbers():
    data = _compute()
    html = build_report_page(data)

    assert "<!doctype html>" in html.lower()
    assert "Temporal Transformer" in html
    assert "XGBoost" in html
    assert "Granger" in html
    assert "Spearman" in html
    assert "confound" in html.lower() or "kısmi korelasyon" in html.lower()
    # Ana sayfaya dönüş linki olmalı.
    assert 'href="index.html"' in html
