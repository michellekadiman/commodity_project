"""Diebold-Mariano test for equal predictive accuracy.

Convention used throughout: the loss differential is
    d_t = loss(candidate)_t - loss(baseline)_t,
so a NEGATIVE DM statistic means the candidate architecture forecasts
BETTER than the baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def dm_test(
    e_candidate: np.ndarray,
    e_baseline: np.ndarray,
    h: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """DM test with Newey-West long-run variance and the Harvey-Leybourne-
    Newbold small-sample correction. Returns (statistic, two-sided p-value).

    Although under the null 1-step-ahead loss differentials are serially
    uncorrelated, volatility clustering in daily returns induces strong
    autocorrelation in squared errors, so we use a Bartlett-kernel HAC
    variance with the standard n^(1/3) truncation rather than lag 0.
    """
    e1 = np.asarray(e_candidate, dtype=float)
    e2 = np.asarray(e_baseline, dtype=float)
    ok = np.isfinite(e1) & np.isfinite(e2)
    d = np.abs(e1[ok]) ** power - np.abs(e2[ok]) ** power
    n = d.size
    if n < 10:
        return np.nan, np.nan

    d_bar = d.mean()
    dc = d - d_bar
    lag = int(np.floor(n ** (1 / 3)))
    lrv = float(dc @ dc) / n
    for l in range(1, lag + 1):
        gamma = float(dc[l:] @ dc[:-l]) / n
        lrv += 2.0 * (1.0 - l / (lag + 1)) * gamma
    if lrv <= 0:
        return np.nan, np.nan

    stat = d_bar / np.sqrt(lrv / n)
    # HLN correction (negligible at n ~ 4800 but included for correctness).
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    pval = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(pval)


def dm_tables(
    errors: dict[str, pd.DataFrame],
    baseline: str = "VanillaAE",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-commodity DM stats and p-values for every architecture vs baseline.

    Rows: architectures (baseline excluded). Columns: commodities.
    """
    base_err = errors[baseline]
    names = [n for n in errors if n != baseline]
    stats_df = pd.DataFrame(index=names, columns=base_err.columns, dtype=float)
    pvals_df = pd.DataFrame(index=names, columns=base_err.columns, dtype=float)
    for name in names:
        cand = errors[name].reindex(base_err.index)
        for col in base_err.columns:
            s, p = dm_test(cand[col].to_numpy(), base_err[col].to_numpy())
            stats_df.loc[name, col] = s
            pvals_df.loc[name, col] = p
    return stats_df, pvals_df


def pooled_dm(
    e_candidate: pd.DataFrame,
    e_baseline: pd.DataFrame,
) -> tuple[float, float]:
    """Panel-level DM: average the squared-error differential across
    commodities each day, then run a single DM test on that series. This
    collapses the cross-section to one time series, so the test variance
    automatically reflects cross-sectional dependence between commodities
    (which a naive average of 22 per-commodity tests would ignore).
    """
    d = (e_candidate.to_numpy() ** 2 - e_baseline.to_numpy() ** 2).mean(axis=1)
    # Reuse dm_test machinery via a synthetic decomposition: pass sqrt of
    # shifted losses is fragile; instead inline the same HAC computation.
    n = d.size
    d_bar = d.mean()
    dc = d - d_bar
    lag = int(np.floor(n ** (1 / 3)))
    lrv = float(dc @ dc) / n
    for l in range(1, lag + 1):
        gamma = float(dc[l:] @ dc[:-l]) / n
        lrv += 2.0 * (1.0 - l / (lag + 1)) * gamma
    if lrv <= 0:
        return np.nan, np.nan
    stat = d_bar / np.sqrt(lrv / n)
    stat *= np.sqrt((n - 1) / n)  # HLN with h = 1
    pval = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(pval)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH-FDR rejection mask. Used when counting 'significant' commodities
    per architecture, since 22 tests per architecture would otherwise
    overstate significance.
    """
    p = np.asarray(pvals, dtype=float)
    mask = np.zeros(p.shape, dtype=bool)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return mask
    pv = p[ok]
    order = np.argsort(pv)
    m = pv.size
    thresh = alpha * (np.arange(1, m + 1)) / m
    below = pv[order] <= thresh
    if below.any():
        k = np.max(np.nonzero(below)[0])
        rej = np.zeros(m, dtype=bool)
        rej[order[: k + 1]] = True
        mask[ok] = rej
    return mask
