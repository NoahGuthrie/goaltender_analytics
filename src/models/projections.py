"""
Kalman filter projection system for goaltender performance.

Uses a linear Gaussian state-space model to track latent "true talent"
over time and project future performance.

State-space formulation:
    Transition:   talent_t = talent_{t-1} + process_noise
    Observation:  observed_GSAx2_t = talent_t + measurement_noise

Features:
    - Separates signal (true talent change) from noise (seasonal variance)
    - Handles missing data (injuries, lockout) naturally
    - Integrates aging curves as time-varying drift
    - Outputs posterior distributions, not just point estimates

Benchmarked against:
    - Naive (previous season GSAx)
    - 3-year weighted average
    - Previous season save percentage
"""

import logging

logger = logging.getLogger(__name__)


# TODO: Implement in Phase 3
# - Estimate process noise from historical career data
# - Estimate measurement noise from within-season GSAx variance
# - Build Kalman filter per goalie
# - Integrate aging curve as drift term
# - Generate projections with confidence intervals
# - Backtesting (train 2007-2020, predict 2021-2025)
