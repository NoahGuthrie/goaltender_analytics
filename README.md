<div align="center">

# 🥅 Goaltender Analytics

**A next-generation goaltender evaluation system for the NHL.**

*Rethinking how we measure, compare, and project NHL goaltenders — because save percentage isn't good enough.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

[**Methodology**](#methodology) · [**Metrics**](#novel-metrics) · [**Data**](#data-pipeline) · [**Dashboard**](#dashboard) · [**Contributing**](#contributing)

</div>

## The Problem

Current goaltender evaluation is broken. The industry standard — **Goals Saved Above Expected (GSAx)** — relies on expected goals models that:

- ❌ **Ignore pre-shot context** — a shot after a cross-ice pass is far harder to save than the same shot from a set position, but existing models treat them identically
- ❌ **Can't measure rebound control** — a goalie who gives up dangerous rebounds creates goals that get attributed to the next shot, not to the goalie who caused the chaos
- ❌ **Conflate goalie skill with team defense** — a goalie behind Carolina's structure looks elite; the same goalie behind a porous defense looks average
- ❌ **Barely predict the future** — GSAx correlates with itself at r ≈ 0.35–0.45 year-over-year. Teams making $8M/year decisions with this data are essentially guessing.

## The Solution

This project builds a **complete goaltender evaluation pipeline** from the ground up:

1. **An enhanced expected goals model** trained on 19 seasons of NHL data with 15+ features that existing public models don't use — including pre-shot event sequences, cross-ice movement indicators, and estimated screening
2. **Four novel metrics** that address the specific failures of existing goaltender evaluation
3. **A Bayesian framework** that separates goalie talent from team system effects
4. **A Kalman filter projection system** that outperforms naive baselines at predicting future performance

## Novel Metrics

### GSAx 2.0 — Enhanced Goals Saved Above Expected
Same formula as traditional GSAx, but powered by a dramatically better xG model. The enhanced model incorporates pre-shot sequences, lateral movement demand, and situational context that existing public models ignore.

### RCI — Rebound Control Index
*Nobody publishes this systematically.* Measures how often a goalie generates rebounds, how dangerous those rebounds are, and combines them into a single score. A goalie with elite save percentage but terrible rebound control is a ticking time bomb — this metric identifies that risk.

### MDA — Movement Demand Adjustment
Quantifies the lateral and positional work a goalie had to do before each save. Adjusts evaluation for the difficulty of the saves faced, not just the shots. A goalie making spectacular cross-crease saves shouldn't be penalized when they occasionally miss one.

### DSIS — Defensive System Impact Score
Uses **Bayesian hierarchical modeling** to separate goaltender talent from team defensive effects. Leverages goalie team-changes as natural experiments. Instead of a single number, produces posterior distributions with credible intervals — honestly communicating how confident we should be.

## Architecture

```
goaltender_analytics/
├── src/
│   ├── ingestion/         # NHL API data scrapers (play-by-play, shifts, EDGE)
│   ├── features/          # Feature engineering pipeline
│   ├── models/            # xG model training, Bayesian models, Kalman filter
│   ├── metrics/           # Novel metric computation (GSAx 2.0, RCI, MDA, DSIS)
│   └── viz/               # Visualization & player card generation
├── dashboard/             # Interactive web dashboard (Vite + D3.js)
├── notebooks/             # Exploratory analysis & methodology development
├── tests/                 # Unit & integration tests
├── docs/                  # Methodology writeups
└── data/                  # Local data storage (not tracked by git)
    ├── raw/               # Immutable API responses (Parquet)
    ├── processed/         # Feature-engineered tables
    └── models/            # Serialized trained models
```

## Tech Stack

| Component | Technology | Rationale |
|:---|:---|:---|
| **Language** | Python 3.11+ | Industry standard for data science; richest ML ecosystem |
| **Data Storage** | DuckDB + Parquet | Columnar analytics engine — 10-50x faster than row-oriented DBs for our workload. Zero server overhead. |
| **xG Model** | CatBoost | Native categorical feature handling (shot types, event sequences, arenas). Oblivious trees resist overfitting on our imbalanced 8% goal rate. Robust defaults. |
| **Bayesian Modeling** | PyMC | Hierarchical model for DSIS. Full posterior distributions with credible intervals. JAX backend for performance. |
| **Projections** | Kalman Filter (pykalman) | Optimal recursive estimator for separating signal from noise in volatile goalie performance. Handles missing data naturally. |
| **Visualization** | matplotlib + SHAP + D3.js | Static player cards (matplotlib), model interpretability (SHAP), interactive dashboard (D3.js) |
| **Dashboard** | Vite + FastAPI | Lightweight, modern frontend with Python API serving pre-computed metrics |
| **Package Management** | uv | Faster, more reliable than pip. Modern Python packaging. |

## Data Pipeline

```
NHL API (play-by-play)  ──→  JSON  ──→  Parquet (raw/)
NHL API (shift charts)  ──→  JSON  ──→  Parquet (raw/)
NHL EDGE (scraped)      ──→  JSON  ──→  Parquet (raw/)
                                              │
                                    Feature Engineering
                                              │
                                    Parquet (processed/)
                                              │
                              ┌────────────────┼─────────────────┐
                              │                │                 │
                        CatBoost xG       PyMC DSIS       Kalman Filter
                              │                │                 │
                              └────────────────┼─────────────────┘
                                              │
                                    Metric Computation
                                    (GSAx 2.0, RCI, MDA, DSIS)
                                              │
                                ┌─────────────┼─────────────────┐
                                │             │                 │
                          Player Cards   Dashboard API   Methodology Writeup
```

**Coverage:** 19 NHL seasons (2007–08 through 2025–26) · ~24,000+ games · ~500,000+ shots on goal

## Methodology

### xG Model Evaluation

| Metric | Purpose |
|:---|:---|
| **Log Loss** | Primary — measures calibration and discrimination jointly |
| **Brier Score** | Mean squared error of probability forecasts |
| **AUC-ROC** | Discrimination — can the model rank shots by danger? |
| **Calibration Curve** | Do predicted probabilities match observed goal frequencies? |

**Validation:** Leave-One-Season-Out cross-validation to prevent temporal leakage. Post-hoc isotonic regression calibration.

### Interpretability
Every model ships with **SHAP values** for global and local explanations. If a coach can't understand *why* the model says what it says, the model is useless.

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/NoahGuthrie/goaltender_analytics.git
cd goaltender_analytics

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Or with pip
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Ingest data (this takes a while on first run — ~15 hours for full history)
python -m src.ingestion.scrape_pbp --seasons 2023 2024 2025

# 2. Engineer features
python -m src.features.build_features

# 3. Train the xG model
python -m src.models.train_xg

# 4. Compute goaltender metrics
python -m src.metrics.compute_all

# 5. Generate player cards
python -m src.viz.player_cards --season 2025
```

## Roadmap

- [x] Project structure & infrastructure
- [ ] NHL API data ingestion pipeline (play-by-play + shifts)
- [ ] Feature engineering pipeline
- [ ] Baseline xG model (logistic regression)
- [ ] Enhanced xG model (CatBoost with full feature set)
- [ ] GSAx 2.0 metric computation
- [ ] Rebound Control Index (RCI)
- [ ] Movement Demand Adjustment (MDA)
- [ ] Defensive System Impact Score (DSIS) — Bayesian hierarchical model
- [ ] Kalman filter projection system
- [ ] Goalie player cards (static images)
- [ ] Interactive web dashboard
- [ ] NHL EDGE data scraper (Advanced Tracking Metrics)
- [ ] Methodology writeups
- [ ] Backtesting report

## Contributing

Contributions, ideas, and feedback are welcome. If you're interested in hockey analytics and want to collaborate:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

**Noah Guthrie** — [GitHub](https://github.com/NoahGuthrie)

---

<div align="center">

*Built with the belief that goaltender evaluation deserves better than save percentage.*

</div>
