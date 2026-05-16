from __future__ import annotations

from cfpipeline.factors import (
    BASE_FACTOR_COLS,
    compute_factors,
    factor_corr_matrix,
    rank_ic_by_month,
    standardize_cross_section,
)


def test_factor_construction_and_ic(make_daily_sample) -> None:
    daily = make_daily_sample(symbols=6)
    factors = compute_factors(daily, include_enhanced=True)
    assert not factors.empty
    for col in [*BASE_FACTOR_COLS, "REVERSAL_1M", "VOL_1M", "ILLIQUIDITY"]:
        assert col in factors.columns

    factors = standardize_cross_section(factors, BASE_FACTOR_COLS)
    for col in ["z_VALUE", "z_MOM_12_1", "z_QUALITY", "z_SIZE"]:
        assert col in factors.columns
        assert factors[col].notna().any()

    ic = rank_ic_by_month(factors, ["z_VALUE", "z_SIZE"])
    assert set(ic.columns) == {"month_end", "factor", "IC"}
    assert not ic.empty
    corr = factor_corr_matrix(factors, BASE_FACTOR_COLS)
    assert corr.shape == (4, 4)
