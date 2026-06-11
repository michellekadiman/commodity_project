"""Synthetic-data tests for every module.

Run with either:
    python tests/test_synthetic.py
    pytest tests/test_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.autoencoders import ALL_ARCHITECTURES, VanillaAE, RecurrentAE, MaskedAE
from src.evaluation.dm_test import benjamini_hochberg, dm_test, pooled_dm
from src.evaluation.rolling import derive_seed, run_rolling_experiment

N, K, T = 22, 5, 300


def make_factor_panel(T: int = T, n: int = N, k: int = 3, seed: int = 0) -> np.ndarray:
    """Panel with a true low-rank factor structure plus idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, k))
    loadings = rng.standard_normal((k, n))
    return factors @ loadings + 0.5 * rng.standard_normal((T, n))


def test_all_architectures_interface():
    X = make_factor_panel()
    X = (X - X.mean(0)) / X.std(0)
    for cls in ALL_ARCHITECTURES:
        model = cls(n_inputs=N, n_factors=K, epochs=30, seed=0)
        assert model.get_name() == cls.name
        model.fit(X)
        Z = model.encode(X)
        assert Z.shape == (T, K), f"{cls.name}: bad shape {Z.shape}"
        assert np.isfinite(Z).all(), f"{cls.name}: non-finite factors"
    print("ok: all 8 architectures satisfy fit/encode/get_name with (T, K) output")


def test_determinism():
    X = make_factor_panel()
    a = VanillaAE(n_inputs=N, n_factors=K, epochs=30, seed=7)
    b = VanillaAE(n_inputs=N, n_factors=K, epochs=30, seed=7)
    a.fit(X), b.fit(X)
    assert np.allclose(a.encode(X), b.encode(X)), "same seed must give same factors"
    print("ok: identical seeds reproduce identical factors")


def test_recurrent_is_causal():
    """Perturbing future observations must not change past factors."""
    X = make_factor_panel(seed=1)
    model = RecurrentAE(n_inputs=N, n_factors=K, epochs=20, seed=0)
    model.fit(X)
    Z1 = model.encode(X)
    X_perturbed = X.copy()
    X_perturbed[200:] += 10.0
    Z2 = model.encode(X_perturbed)
    assert np.allclose(Z1[:200], Z2[:200], atol=1e-5), "RecurrentAE leaks the future"
    assert not np.allclose(Z1[200:], Z2[200:]), "perturbation should change factors"
    print("ok: RecurrentAE factors are causal (no look-ahead)")


def test_masked_mask_shape():
    model = MaskedAE(n_inputs=N, n_factors=K, epochs=5, seed=0)
    import torch

    torch.manual_seed(0)
    mask = model._sample_mask(252, N, "cpu")
    frac = mask.float().mean().item()
    assert 0.05 < frac < 0.5, f"mask fraction {frac} out of range"
    # Contiguity: masked runs in each column should be >= 2 days on average.
    col = mask[:, 0].numpy().astype(int)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, col, 0])))[::2]
    if runs.size:
        assert runs.mean() >= 2
    print(f"ok: MaskedAE contiguous block masking (masked frac = {frac:.2f})")


def test_dm():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(2000)
    # Identical accuracy: stat near 0.
    s, p = dm_test(base + 0.0, base.copy())
    assert np.isnan(s) or abs(s) < 1e-9
    # Candidate clearly worse: significantly positive stat.
    worse = base + rng.standard_normal(2000) * 0.8
    s, p = dm_test(worse, base)
    assert s > 1.96 and p < 0.05, f"expected significant deterioration, got {s=}, {p=}"
    # Candidate clearly better: negative stat.
    s, p = dm_test(base * 0.5, base)
    assert s < -1.96 and p < 0.05
    # Pooled DM and BH-FDR sanity.
    df_b = pd.DataFrame(rng.standard_normal((500, 4)))
    df_c = df_b * 0.5
    s, p = pooled_dm(df_c, df_b)
    assert s < 0 and p < 0.05
    mask = benjamini_hochberg(np.array([0.001, 0.2, 0.8, 0.01]))
    assert mask[0] and not mask[2]
    print("ok: DM test, pooled DM and BH-FDR behave as expected")


def test_rolling_pipeline():
    X = make_factor_panel(T=180, seed=2)
    idx = pd.bdate_range("2020-01-01", periods=180)
    panel = pd.DataFrame(X, index=idx, columns=[f"C{i}" for i in range(N)])
    factory = lambda seed: VanillaAE(n_inputs=N, n_factors=K, epochs=15, seed=seed)
    res = run_rolling_experiment(panel, factory, window=120, refit_every=21, base_seed=0)
    err = res["errors"]
    assert err.shape[1] == N and len(err) == 180 - 120
    assert np.isfinite(err.to_numpy()).all()
    assert err.index[0] == idx[120]
    # Same config must reproduce identical errors (seeding is deterministic).
    res2 = run_rolling_experiment(panel, factory, window=120, refit_every=21, base_seed=0)
    assert np.allclose(err.to_numpy(), res2["errors"].to_numpy())
    assert derive_seed(0, "A", 10) != derive_seed(0, "B", 10)
    print("ok: rolling pipeline produces aligned, finite, reproducible OOS errors")


if __name__ == "__main__":
    test_all_architectures_interface()
    test_determinism()
    test_recurrent_is_causal()
    test_masked_mask_shape()
    test_dm()
    test_rolling_pipeline()
    print("\nAll synthetic tests passed.")
