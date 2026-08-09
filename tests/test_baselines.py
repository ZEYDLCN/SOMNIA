from src.models.baselines import run


def test_baselines_run_and_beat_naive():
    results = run(window_size=7)

    assert set(results) == {"naive", "linear_regression", "xgboost"}
    for name, m in results.items():
        for split in ("val", "test"):
            assert m[split]["mae"] > 0
            assert m[split]["rmse"] >= m[split]["mae"]

    # XGBoost, küçük veri setinde bile naive'i geçmeli (test seti).
    assert results["xgboost"]["test"]["mae"] < results["naive"]["test"]["mae"]
