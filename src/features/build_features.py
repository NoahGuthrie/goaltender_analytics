"""
Feature engineering pipeline.

Constructs the full feature matrix for the enhanced xG model from raw
play-by-play and shift data. Features include:

Standard (baseline):
    - Shot distance, angle, type, strength state, empty net, period, home/away

Pre-shot context (novel):
    - Previous event type and 2-event lookback sequence
    - Time since previous event (goalie preparedness proxy)
    - Cross-ice pass indicator (lateral movement demand)
    - Distance of previous event from net (puck origin)
    - Shot sequence number (rebound pressure)

Environmental (novel):
    - Shooting team zone time (30s window — sustained pressure proxy)
    - Goalie shift duration at time of shot (fatigue proxy)
    - Estimated screen indicator (skaters in shooting lane)
    - Nearest defender distance to shooter
    - Score differential, arena (recording bias correction)
"""

import logging

logger = logging.getLogger(__name__)


# TODO: Implement in Phase 1
# - Load raw play-by-play Parquet files
# - Sequence construction (event lookback)
# - Cross-ice pass detection (y-coordinate sign flip)
# - Zone time calculation (rolling 30s window)
# - Shift cross-reference for goalie duration
# - Screen estimation from on-ice player positions
# - Arena bias detection and flagging
# - Output: feature matrix as Parquet (processed/)
