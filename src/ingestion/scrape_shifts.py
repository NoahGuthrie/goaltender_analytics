"""
NHL API shift chart scraper.

Pulls player shift data (start/end times per shift per player per game)
from the NHL stats API. This data is essential for:
- Determining which players are on the ice at the time of each shot
- Calculating goalie shift duration at the moment of each shot
- Estimating screening (counting skaters in shooting lanes)

Endpoint:
    https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}
"""

import logging

logger = logging.getLogger(__name__)


# TODO: Implement in Phase 0
# - Shift chart fetcher (get all shifts for a single game)
# - Cross-reference with play-by-play timestamps
# - On-ice player reconstruction (who was on ice at time T?)
# - Parquet writer (partitioned by season)
