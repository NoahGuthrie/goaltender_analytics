"""
Goaltender player card generator.

Produces shareable PNG images summarizing a goaltender's performance
across all novel metrics. Designed for Twitter/X sharing and blog posts.

Card contents:
    - Player name, team, headshot
    - GSAx 2.0 (season + career)
    - Rebound Control Index (percentile bar)
    - Movement Demand Adjustment (percentile bar)
    - Defensive System Impact Score (with credible interval)
    - Save percentage by shot zone (heatmap overlay on net)
    - Projection arrow (trending up/down/stable)

Style: dark theme, clean typography, branded color palette.
"""

import logging

logger = logging.getLogger(__name__)


# TODO: Implement in Phase 4
# - Card layout design (matplotlib figure with gridspec)
# - Metric percentile computation (league-wide context)
# - Net heatmap overlay (save % by zone)
# - Projection indicator from Kalman filter
# - Batch generation for all active goalies
# - Export as high-res PNG
