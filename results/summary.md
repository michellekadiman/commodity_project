# Autoencoder architecture comparison — summary

Out-of-sample one-step-ahead forecast accuracy of factor-augmented AR(1) models, rolling 252-day windows, monthly refits. Baseline: VanillaAE.

Ranking (by pooled OOS MSE ratio vs. baseline; lower is better):

| architecture   |   pooled_mse |   pooled_mse_ratio |   median_commodity_mse_ratio |   n_commodities_better |   n_sig_better_fdr |   n_sig_worse_fdr |   median_dm_stat |   pooled_dm_stat |   pooled_dm_pvalue |
|:---------------|-------------:|-------------------:|-----------------------------:|-----------------------:|-------------------:|------------------:|-----------------:|-----------------:|-------------------:|
| ContractiveAE  |       1.0104 |             0.9983 |                       0.9979 |                     14 |                  0 |                 0 |          -0.5596 |          -1.2974 |             0.1945 |
| RecurrentAE    |       1.011  |             0.9989 |                       0.999  |                     12 |                  0 |                 0 |          -0.2231 |          -0.7353 |             0.4622 |
| MaskedAE       |       1.0111 |             0.999  |                       0.9986 |                     12 |                  0 |                 0 |          -0.2933 |          -0.6097 |             0.5421 |
| DenoisingAE    |       1.0112 |             0.9991 |                       0.9993 |                     13 |                  0 |                 0 |          -0.1448 |          -0.722  |             0.4703 |
| VAE            |       1.0115 |             0.9994 |                       0.9997 |                     12 |                  0 |                 0 |          -0.0632 |          -0.3944 |             0.6933 |
| SparseAE       |       1.0119 |             0.9998 |                       1.0009 |                      9 |                  0 |                 0 |           0.1756 |          -0.1509 |             0.88   |
| VanillaAE      |       1.0121 |             1      |                       1      |                      0 |                nan |               nan |         nan      |         nan      |           nan      |
| BetaVAE        |       1.0124 |             1.0003 |                       1.0006 |                     10 |                  0 |                 0 |           0.1067 |           0.1627 |             0.8708 |

Best-ranked architecture: **ContractiveAE**.

Selection guidance: prefer an architecture whose advantage is (1) significant in the pooled DM test, (2) broad across commodities (`n_commodities_better`, `n_sig_better_fdr`), and (3) stable over time (see `cumulative_advantage.png`). Per-commodity details are in `mse_per_commodity.csv`, `dm_statistics.csv`, `dm_pvalues.csv`.