# Hong Kong COVID-19 SEIHDR Model

An age-stratified SEIHDR compartmental model fitted to Hong Kong government
COVID-19 data, used to study how serious-case and hospitalization dynamics
changed across major epidemic waves.

The population is split into three age groups — **children (0-20)**,
**adults (21-64)**, and **elders (65+)** — and the model is calibrated
separately against **wave 4 (Ancestral strain)** and **wave 5 (Omicron)**.
The central question is how public health and social measures affected the
effective transmission rate (R₀) under Alpha/Omicron versus the earlier
ancestral strain.

## Repository layout

| Path | Purpose |
|------|---------|
| `model/SEIHDR.py` | The age-stratified SEIHDR model (S, E, I, H, D, R compartments) with parameters for the ancestral and Omicron strains. |
| `model/matrixgenerator.py` | Aggregates the raw 15x15 age-contact matrix into the 3x3 matrix used by the model. |
| `model/fit.py` | Calibrates transmission (β) and the initial seed fraction against observed critical-case data, and plots the resulting fit and R₀ trajectory. |
| `data/plot.py` | Loads and plots the Hong Kong government series (critical hospitalizations or deaths). |
| `data/HKGov.csv` | The dataset used for the analysis. |

## Getting started

Install the dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```

### Plot the source data

```
python data/plot.py                       # critical cases, custom date range
python data/plot.py --wave 5              # critical cases during Omicron
python data/plot.py --graph death --wave 4
```

Options: `--graph {critical,death}`, `--wave {4,5,custom}`,
`--start YYYY-MM-DD`, `--end YYYY-MM-DD`.

### Fit the model

```
python model/fit.py --wave 4 --how rss-rough
python model/fit.py --wave 5 --splines y --how rss-rough
```

Options: `--wave {4,5}`, `--how {rss,rss-rough}`, `--splines {y,n}`.

The calibrator co-optimizes the initial seed fraction and the transmission
rate β. With `--splines y`, β is a 3-knot linear spline (knots placed
adaptively on the cumulative data) instead of a constant. A roughness penalty
and a late-wave growth penalty keep the fitted curve from overfitting the
tail of the epidemic. The script saves PNGs of each fit to
`model/fitted_plots/` (constant β) or `model/fitted_plots_splines/` (splines).

## Model

The SEIHDR model tracks, per age group: **S**usceptible, **E**xposed,
**I**nfected, **H**ospitalized (critical), **D**ead, and **R**ecovered.
Transmission follows the aggregated contact matrix, and age-group parameters
(case-to-critical rate α, recovery rates τ₁/τ₂, death rates δ₁/δ₂) are taken
from the literature for each strain. The ratio R₀ = β × infectious period is
reported alongside every fit.

## Data

All data is from the Hong Kong government's official open dataset on
COVID-19 hospital and testing data:
[data.gov.hk](https://data.gov.hk/en-data/dataset/hk-dh-chpsebcddr-novel-infectious-agent/resource/9252c845-3aea-4ea7-abae-b385916106b3)

## License

MIT — see [LICENSE](LICENSE).
