"""
NHL EDGE tracking data scraper.

Scrapes advanced tracking metrics from the NHL EDGE platform:
- Skating speed (max, burst frequencies)
- Distance traveled per game/shift
- Acceleration/deceleration patterns

Note: There is no official public API for EDGE data. This scraper
interacts with the NHL's web-facing data endpoints, which may change
without notice. The core xG model is designed to function without EDGE
data (shorter history window) as a fallback.

Available from: 2021-22 season onward.
"""

import logging

logger = logging.getLogger(__name__)


# TODO: Implement in Phase 0
# - EDGE endpoint discovery and documentation
# - Per-player per-game skating metrics fetcher
# - Fallback handling if endpoints change
# - Parquet writer
