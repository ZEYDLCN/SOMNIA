from src.analysis.causality import (
    GRANGER_CANDIDATES,
    _load_df,
    caffeine_timing_natural_experiment,
    partial_correlation,
    run_granger_battery,
)


def test_partial_correlation_shapes():
    df = _load_df()
    result = partial_correlation(df, "caffeine_mg", "sleep_quality", "stress_score")
    assert -1.0 <= result["raw_pearson_r"] <= 1.0
    assert -1.0 <= result["partial_r_controlling_for_control"] <= 1.0
    assert 0.0 <= result["raw_p_value"] <= 1.0


def test_caffeine_natural_experiment_groups_are_disjoint_and_cover_all_rows():
    df = _load_df()
    result = caffeine_timing_natural_experiment(df)
    assert result["n_close_to_bed"] + result["n_far_from_bed"] == len(df)
    assert result["n_close_to_bed"] > 0
    assert result["n_far_from_bed"] > 0


def test_granger_battery_covers_all_candidates():
    df = _load_df()
    results = run_granger_battery(df)
    features_tested = {r["feature"] for r in results}
    assert features_tested == set(GRANGER_CANDIDATES)
    for r in results:
        if "error" in r:
            continue
        assert 0.0 <= r["best_p_value"] <= 1.0
        assert r["best_lag"] in r["p_value_per_lag"]
