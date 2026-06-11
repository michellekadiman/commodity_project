"""Rolling-window factor-augmented AR(1) forecasting experiment.

Scheme (per architecture):

  - Training window: 252 trading days (~1 year), rolling.
  - Refit frequency: every 21 trading days (~1 month). Justification:
      (a) factor loadings in commodity panels evolve slowly relative to a
          month, so daily refits buy little signal;
      (b) each AE fit lands in a different local optimum, so daily refits
          inject optimizer noise into the factor series and contaminate the
          *architecture* comparison -- fewer, deterministic refits give a
          cleaner contrast;
      (c) compute: monthly refits make the 8-architecture x ~230-window grid
          tractable (~1,800 AE fits instead of ~38,000).
  - Forecasts remain DAILY: within each 21-day block the frozen encoder and
    frozen OLS produce a one-step-ahead forecast every day.
  - Standardization: mean/std fit on the training window only; the same
    scaler transforms the OOS observations of that block (no look-ahead).
  - Forecast model: per commodity i, OLS of r_{i,t+1} on [1, r_{i,t}, F_t]
    where F_t are the K autoencoder factors -- a factor-augmented AR(1).
  - Errors are reported in (de-standardized) return units so MSE is
    comparable across architectures and windows.

No look-ahead: F_t is computed from data up to and including day t (the
RecurrentAE encoder is causal by construction), and forecasts target t+1.
"""

from __future__ import annotations

import zlib
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.autoencoders.base import BaseAutoencoder


def derive_seed(base_seed: int, arch_name: str, window_start: int) -> int:
    """Deterministic per-(architecture, window) seed.

    Results are reproducible and independent of the order in which
    architectures or windows are executed.
    """
    h = zlib.crc32(f"{arch_name}|{window_start}".encode())
    return int((base_seed * 1_000_003 + h) % (2**31 - 1))


def run_rolling_experiment(
    returns: pd.DataFrame,
    model_factory: Callable[[int], BaseAutoencoder],
    window: int = 252,
    refit_every: int = 21,
    base_seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the rolling experiment for one architecture.

    Returns dict with 'errors' and 'forecasts' DataFrames (OOS rows only,
    indexed by date, columns = commodities).
    """
    X = returns.to_numpy(dtype=np.float64)
    T, N = X.shape
    if T <= window + 1:
        raise ValueError("Not enough observations for the chosen window.")

    forecasts = np.full((T, N), np.nan)
    arch_name = model_factory(0).get_name()

    refit_points = list(range(window, T, refit_every))
    for step, s in enumerate(refit_points):
        block_end = min(s + refit_every, T)  # forecast targets: s .. block_end-1

        Xtr = X[s - window : s]
        mu = Xtr.mean(axis=0)
        sd = np.maximum(Xtr.std(axis=0), 1e-8)
        Ztr = (Xtr - mu) / sd

        model = model_factory(derive_seed(base_seed, arch_name, s))
        model.fit(Ztr)

        # Encode window + OOS block in a single causal pass. For pointwise
        # encoders this equals row-wise encoding; for the RecurrentAE it
        # gives each OOS day its full (causal) history from the window start.
        Zall = (X[s - window : block_end] - mu) / sd
        F = model.encode(Zall)  # (window + block_len - ?, K)
        Ftr = F[:window]

        # Factor-augmented AR(1) per commodity, fit on the training window
        # only (same information set as the autoencoder fit).
        intercepts = np.empty(N)
        own_coef = np.empty(N)
        factor_coef = np.empty((N, F.shape[1]))
        for i in range(N):
            Xreg = np.column_stack([Ztr[:-1, i], Ftr[:-1]])
            reg = LinearRegression().fit(Xreg, Ztr[1:, i])
            intercepts[i] = reg.intercept_
            own_coef[i] = reg.coef_[0]
            factor_coef[i] = reg.coef_[1:]

        # Daily one-step-ahead forecasts within the block.
        origins = np.arange(s - 1, block_end - 1)       # forecast made at t
        rows = origins - (s - window)                    # index into Zall/F
        z_hat = intercepts + own_coef * Zall[rows] + F[rows] @ factor_coef.T
        forecasts[origins + 1] = mu + sd * z_hat

        if progress is not None:
            progress(step + 1, len(refit_points))

    oos = ~np.isnan(forecasts).all(axis=1)
    fc = pd.DataFrame(forecasts[oos], index=returns.index[oos], columns=returns.columns)
    actual = returns.loc[fc.index]
    return {"forecasts": fc, "errors": actual - fc}
