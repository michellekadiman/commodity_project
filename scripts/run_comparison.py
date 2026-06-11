#!/usr/bin/env python3
"""Entry point: systematic autoencoder comparison for commodity forecasting.

Usage:
    python scripts/run_comparison.py                    # full run
    python scripts/run_comparison.py --quick            # fast smoke test
    python scripts/run_comparison.py --archs VanillaAE VAE

Note: the panel lives at data/returns.csv in this repository (T=5075, N=22
daily log-returns); pass --data to point elsewhere.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.autoencoders import ALL_ARCHITECTURES  # noqa: E402
from src.evaluation.reporting import write_all_outputs  # noqa: E402
from src.evaluation.rolling import run_rolling_experiment  # noqa: E402

ARCH_BY_NAME = {cls.name: cls for cls in ALL_ARCHITECTURES}
BASELINE = "VanillaAE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "returns.csv"))
    p.add_argument("--output", default=str(REPO_ROOT / "results"))
    p.add_argument("--window", type=int, default=252, help="training window (days)")
    p.add_argument("--refit-every", type=int, default=21, help="AE refit cadence (days)")
    p.add_argument("--factors", type=int, default=5, help="latent dimension K")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--archs", nargs="+", default=list(ARCH_BY_NAME),
                   choices=list(ARCH_BY_NAME), help="subset of architectures")
    p.add_argument("--skip-existing", action="store_true",
                   help="reuse per-architecture error files already in results/errors/")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: last ~3y of data, fewer epochs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if BASELINE not in args.archs:
        args.archs = [BASELINE] + args.archs  # DM tests need the baseline

    returns = pd.read_csv(args.data, index_col=0, parse_dates=True)
    assert not returns.isna().any().any(), "panel must be clean"
    if args.quick:
        returns = returns.iloc[-756:]
        args.epochs = 60

    out_dir = Path(args.output)
    err_dir = out_dir / "errors"
    err_dir.mkdir(parents=True, exist_ok=True)

    T, N = returns.shape
    print(f"Panel: T={T}, N={N} | window={args.window}, refit={args.refit_every}, "
          f"K={args.factors}, epochs={args.epochs}, seed={args.seed}")

    errors: dict[str, pd.DataFrame] = {}
    for name in args.archs:
        err_path = err_dir / f"{name}_errors.csv"
        if args.skip_existing and err_path.exists():
            errors[name] = pd.read_csv(err_path, index_col=0, parse_dates=True)
            print(f"[{name}] loaded cached errors from {err_path}")
            continue

        cls = ARCH_BY_NAME[name]
        factory = lambda seed, cls=cls: cls(  # noqa: E731
            n_inputs=N, n_factors=args.factors, epochs=args.epochs, seed=seed
        )
        t0 = time.time()

        def progress(done: int, total: int) -> None:
            if done % 25 == 0 or done == total:
                print(f"  [{name}] window {done}/{total} "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)

        print(f"[{name}] running rolling experiment...")
        result = run_rolling_experiment(
            returns, factory,
            window=args.window, refit_every=args.refit_every,
            base_seed=args.seed, progress=progress,
        )
        errors[name] = result["errors"]
        result["errors"].to_csv(err_path)
        result["forecasts"].to_csv(err_dir / f"{name}_forecasts.csv")
        oos_mse = float((result["errors"].to_numpy() ** 2).mean())
        print(f"[{name}] done in {time.time() - t0:.0f}s | pooled OOS MSE = {oos_mse:.5f}")

    summary = write_all_outputs(errors, out_dir, baseline=BASELINE)
    print("\n=== Architecture ranking (pooled OOS MSE ratio vs. VanillaAE) ===")
    print(summary.round(4).to_string())
    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()
