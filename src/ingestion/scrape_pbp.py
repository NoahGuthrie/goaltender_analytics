"""
NHL API play-by-play scraper.

Pulls game-level event data from api-web.nhle.com for all regular season
and playoff games across specified seasons. Stores raw JSON as Parquet files
partitioned by season for efficient downstream querying.

Supports incremental scraping with checkpoint/resume capability.

Endpoints used:
    - Schedule: https://api-web.nhle.com/v1/schedule/{date}
    - Play-by-play: https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play
"""

import logging

logger = logging.getLogger(__name__)


# TODO: Implement in Phase 0
# - Schedule fetcher (get all game IDs for a season)
# - Play-by-play fetcher (get events for a single game)
# - Rate limiter (1 req/sec with exponential backoff)
# - Checkpoint/resume logic
# - Parquet writer (partitioned by season)
# - Coordinate validation (x ∈ [-100, 100], y ∈ [-42.5, 42.5])
