"""
Metric computation engine.

Orchestrates computation of all four novel goaltender metrics
from the trained xG model and processed data.

Metrics:
    GSAx 2.0 = Σ(enhanced xG per shot faced) - actual goals allowed
    RCI = 1 - (rebound_rate × rebound_danger / league_avg)
    MDA = avg(movement_demand on saves) - avg(movement_demand on goals)
    DSIS = posterior mean from Bayesian hierarchical model

Output: per-goalie per-season metric table (Parquet + CSV)
"""

import logging

logger = logging.getLogger(__name__)


# TODO: Implement in Phase 2
# - Load trained xG model and generate predictions on all shots
# - GSAx 2.0: aggregate enhanced xG by goalie, subtract actual goals
# - RCI: identify rebound sequences, compute sub-metrics
# - MDA: compute angular displacement and time pressure per shot
# - DSIS: load posterior summaries from Bayesian model
# - Combine into unified goalie metric table
# - Export to Parquet and CSV
