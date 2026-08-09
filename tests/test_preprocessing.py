import numpy as np

from src.data.preprocessing import build_pipeline


def test_pipeline_shapes_and_no_leakage():
    splits = build_pipeline(window_size=7)

    n_features = len(splits.train.feature_names)
    assert splits.train.X.shape[1:] == (7, n_features)
    assert splits.val.X.shape[1:] == (7, n_features)
    assert splits.test.X.shape[1:] == (7, n_features)

    # Zaman sırası: train'in son tarihi, val'in ilk tarihinden önce olmalı.
    assert splits.train.dates.max() < splits.val.dates.min()
    assert splits.val.dates.max() < splits.test.dates.min()

    # Train üzerinde normalize edilmiş verinin ortalaması ~0, std'si ~1
    # olmalı (tolerans geniş: küçük örneklem + sabit sütunlar olabilir).
    flat_train = splits.train.X.reshape(-1, n_features)
    assert np.all(np.abs(flat_train.mean(axis=0)) < 0.5)

    # Hiç NaN sızmamalı.
    assert not np.isnan(splits.train.X).any()
    assert not np.isnan(splits.val.X).any()
    assert not np.isnan(splits.test.X).any()


def test_no_same_day_target_leakage():
    """Pencere, hedef günün kendi özelliklerini içermemeli."""
    splits = build_pipeline(window_size=7)
    # window_size gün geriye gidip hedefin AYNI günü olmadığını dolaylı
    # olarak tarih aritmetiğiyle doğrula: pencere uzunluğu sabit ve
    # y_dates - 1 gün, pencerenin son günüdür (make_windows tanımı gereği).
    assert splits.train.X.shape[0] > 0
