# Commodity cleaning provenance log

Automated record of every parsing, alignment, guard, and transform step applied by `clean_commodities.py`.

## Run metadata

| Field | Value |
| --- | --- |
| Generated | 2026-06-01T15:23:48 |
| Source workbook | `raw_data.xlsx` |
| Ticker sheet | `Sheet1` |
| Price sheet | `Sheet3` |
| Outputs | `prices.csv`, `returns.csv` |

---


## Input and block layout

| Metric | Value |
| --- | --- |
| Sheet | Sheet3 |
| Raw shape (rows × cols) | 5329 × 65 |

After dropping fully-empty columns: **44** columns remain (0-based indices: `[0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64]`).

Column count reconciles: **22** `(date, price)` pairs ↔ **22** tickers.


### Ticker reconciliation

Tickers read from `Sheet1` cells `D2:D23`:

`CO1, CL1, NG1, XB1, HO1, GC1, SI1, HG1, LA1, LN1, LX1, KC1, C 1, CT1, LH1, LC1, SB1, S 1, SM1, BO1, KW1, W 1`

Sheet1 tickers match `COMMODITY_MAP` order exactly.


### Per-commodity parse summary

| # | Name | Ticker | Sector | Obs | Start | End | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Brent | CO1 | Energy | 5329 | 2005-10-04 | 2026-06-01 | — |
| 2 | WTI | CL1 | Energy | 5203 | 2005-10-04 | 2026-06-01 | — |
| 3 | NaturalGas | NG1 | Energy | 5202 | 2005-10-04 | 2026-06-01 | — |
| 4 | Gasoline | XB1 | Energy | 5202 | 2005-10-04 | 2026-06-01 | — |
| 5 | Diesel | HO1 | Energy | 5202 | 2005-10-04 | 2026-06-01 | — |
| 6 | Gold | GC1 | Metals | 5203 | 2005-10-04 | 2026-06-01 | — |
| 7 | Silver | SI1 | Metals | 5203 | 2005-10-04 | 2026-06-01 | — |
| 8 | Copper | HG1 | Metals | 5203 | 2005-10-04 | 2026-06-01 | — |
| 9 | Aluminium | LA1 | Metals | 5212 | 2005-10-04 | 2026-05-29 | — |
| 10 | Nickel | LN1 | Metals | 5201 | 2005-10-04 | 2026-05-29 | — |
| 11 | Zinc | LX1 | Metals | 5203 | 2005-10-04 | 2026-05-29 | — |
| 12 | Coffee | KC1 | Agri | 5195 | 2005-10-04 | 2026-06-01 | — |
| 13 | Corn | C 1 | Agri | 5203 | 2005-10-04 | 2026-06-01 | — |
| 14 | Cotton | CT1 | Agri | 5201 | 2005-10-04 | 2026-06-01 | — |
| 15 | LeanHogs | LH1 | Agri | 5204 | 2005-10-04 | 2026-06-01 | — |
| 16 | LiveCattle | LC1 | Agri | 5203 | 2005-10-04 | 2026-06-01 | — |
| 17 | Sugar | SB1 | Agri | 5194 | 2005-10-04 | 2026-06-01 | — |
| 18 | Soybeans | S 1 | Agri | 5203 | 2005-10-04 | 2026-06-01 | — |
| 19 | SoybeanMeal | SM1 | Agri | 5203 | 2005-10-04 | 2026-06-01 | — |
| 20 | SoybeanOil | BO1 | Agri | 5203 | 2005-10-04 | 2026-06-01 | — |
| 21 | HRWWheat | KW1 | Agri | 5203 | 2005-10-04 | 2026-06-01 | — |
| 22 | Wheat | W 1 | Agri | 5203 | 2005-10-04 | 2026-06-01 | — |

Parsed **22** commodity series.


## Calendar alignment

> **Rule:** Outer-join all series on date, then keep only dates where every commodity has a non-missing price (complete-case intersection).

| Step | Result |
| --- | --- |
| After outer join | 5335 rows × 22 cols |
| Rows dropped (any missing price) | 257 |
| Rows retained (intersection) | 5078 |
| Aligned matrix shape | 5078 × 22 |
| Date range | 2005-10-04 → 2026-05-29 |


## Non-positive price guard

> **Rule:** Any price ≤ 0 is replaced with `NaN`; each replacement is logged below.

Applied **before** log-returns so `ln(price)` is never taken on non-positive values.

| Commodity | Date | Original value | Replacement |
| --- | --- | --- | --- |
| WTI | 2020-04-20 | -37.63 | `NaN` |

**Total guard actions:** 1


## Log-returns

Formula per commodity: `r_t = ln(p_t / p_{t-1})`

| Date | Reason | Affected commodities | Action |
| --- | --- | --- | --- |
| 2005-10-04 | First calendar row | All commodities | No prior price for returns |
| 2020-04-20 | NaN log-return | WTI | Row removed from return matrix |
| 2020-04-21 | NaN log-return | WTI | Row removed from return matrix |

| Metric | Value |
| --- | --- |
| Rows before return drop | 5077 |
| Rows dropped (NaN returns) | 2 |
| Final return rows | 5075 |
| Shape (before z-score) | 5075 × 22 |

Return date range: **2005-10-05** → **2026-05-29**


## Per-commodity z-score standardization

> **Rule:** Each column is transformed as `(r - mean) / std` using the final return sample (population std, `ddof=0`).

Full-sample scaling is used here for **representation learning** only. For out-of-sample forecasting, refit the scaler inside each rolling window to avoid look-ahead bias.

| Commodity | Mean (raw returns) | Std (raw returns) |
| --- | --- | --- |
| Brent | 0.000154 | 0.023532 |
| WTI | 0.000180 | 0.026960 |
| NaturalGas | -0.000296 | 0.038775 |
| Gasoline | 0.000169 | 0.026550 |
| Diesel | 0.000162 | 0.022911 |
| Gold | 0.000451 | 0.011800 |
| Silver | 0.000463 | 0.022503 |
| Copper | 0.000263 | 0.017862 |
| Aluminium | 0.000140 | 0.014777 |
| Nickel | 0.000068 | 0.025833 |
| Zinc | 0.000185 | 0.021547 |
| Coffee | 0.000219 | 0.020806 |
| Corn | 0.000161 | 0.018692 |
| Cotton | 0.000075 | 0.018276 |
| LeanHogs | 0.000037 | 0.023887 |
| LiveCattle | 0.000204 | 0.011258 |
| Sugar | 0.000053 | 0.020501 |
| Soybeans | 0.000148 | 0.015125 |
| SoybeanMeal | 0.000137 | 0.018776 |
| SoybeanOil | 0.000240 | 0.016112 |
| HRWWheat | 0.000099 | 0.019709 |
| Wheat | 0.000106 | 0.021223 |

Standardized return matrix shape: **5075 × 22**


## Final outputs

| File | Shape (rows × cols) | Path |
| --- | --- | --- |
| prices.csv | 5078 × 22 | `/Users/shreyanshsharma/Desktop/Resume Projects/Summer Project/explainable_commodity_prices/data/prices.csv` |
| returns.csv | 5075 × 22 | `/Users/shreyanshsharma/Desktop/Resume Projects/Summer Project/explainable_commodity_prices/data/returns.csv` |


### Aligned price panel (post-guard)

| Commodity | Non-null obs | Start | End | NaN count |
| --- | --- | --- | --- | --- |
| Brent | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| WTI | 5077 | 2005-10-04 | 2026-05-29 | 1 |
| NaturalGas | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Gasoline | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Diesel | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Gold | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Silver | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Copper | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Aluminium | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Nickel | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Zinc | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Coffee | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Corn | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Cotton | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| LeanHogs | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| LiveCattle | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Sugar | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Soybeans | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| SoybeanMeal | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| SoybeanOil | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| HRWWheat | 5078 | 2005-10-04 | 2026-05-29 | 0 |
| Wheat | 5078 | 2005-10-04 | 2026-05-29 | 0 |


### Return sample (post row-drop)

All **22** commodities share **5075** return observations from **2005-10-05** → **2026-05-29**.

