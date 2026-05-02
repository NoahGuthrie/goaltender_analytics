"""
NHL API play-by-play scraper.

Pulls game-level event data from api-web.nhle.com for all regular season
and playoff games across specified seasons. Stores raw JSON as Parquet files
partitioned by season for efficient downstream querying.

Supports incremental scraping with checkpoint/resume capability.

Endpoints used:
    - Schedule: https://api-web.nhle.com/v1/schedule/{date}
    - Play-by-play: https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play

Usage:
    python -m src.ingestion.scrape_pbp --seasons 2024 2025
    python -m src.ingestion.scrape_pbp --seasons 2024 --resume --max-games 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.api_client import NHLAPIClient
from src.ingestion.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_PBP_DIR = Path("data/raw/pbp")
RAW_SCHEDULE_DIR = Path("data/raw/schedule")

# NHL season start-year → approximate regular-season start date.
# The schedule API will give us exact dates, but we need a seed date to begin.
# We use Oct 1 of the season start year as a safe starting point (regular
# season always starts in early-to-mid October).
_SEASON_SEED_MONTH_DAY = "10-01"

# Game types we care about: 2 = Regular Season, 3 = Playoffs
_TARGET_GAME_TYPES = {2, 3}

# Game states indicating the game is final
_FINAL_GAME_STATES = {"OFF", "FINAL"}

# All seasons with reliable coordinate data (2007-08 onward)
ALL_SEASONS = list(range(2007, 2026))  # 2007 through 2025 (for 2025-26)


# ---------------------------------------------------------------------------
# Schedule / game-ID discovery
# ---------------------------------------------------------------------------


def discover_game_ids(
    client: NHLAPIClient,
    season: int,
) -> tuple[list[dict], list[int]]:
    """Walk the schedule API to collect all completed game IDs for a season.

    Parameters
    ----------
    client : NHLAPIClient
        The shared API client.
    season : int
        Start year of the season (e.g. ``2024`` for 2024-25).

    Returns
    -------
    game_metadata : list[dict]
        One dict per game with metadata (id, date, venue, teams, type).
    game_ids : list[int]
        Sorted list of game IDs ready for PBP fetching.
    """
    seed_date = f"{season}-{_SEASON_SEED_MONTH_DAY}"
    current_date = seed_date
    season_code = int(f"{season}{season + 1}")

    game_metadata: list[dict] = []
    seen_ids: set[int] = set()

    logger.info("Discovering games for season %d-%d (starting from %s)", season, season + 1, seed_date)

    while current_date is not None:
        schedule = client.get_schedule(current_date)

        for week in schedule.get("gameWeek", []):
            for game in week.get("games", []):
                gid = game["id"]
                game_season = game.get("season", 0)
                game_type = game.get("gameType", 0)
                game_state = game.get("gameState", "")

                # Only games from the target season
                if game_season != season_code:
                    continue

                # Only regular season + playoffs
                if game_type not in _TARGET_GAME_TYPES:
                    continue

                # Only completed games
                if game_state not in _FINAL_GAME_STATES:
                    continue

                # Deduplicate (schedule pages overlap by a week)
                if gid in seen_ids:
                    continue

                seen_ids.add(gid)
                game_metadata.append({
                    "game_id": gid,
                    "season": game_season,
                    "game_type": game_type,
                    "date": week["date"],
                    "venue": game.get("venue", {}).get("default", ""),
                    "home_team_id": game.get("homeTeam", {}).get("id"),
                    "home_team_abbrev": game.get("homeTeam", {}).get("abbrev", ""),
                    "away_team_id": game.get("awayTeam", {}).get("id"),
                    "away_team_abbrev": game.get("awayTeam", {}).get("abbrev", ""),
                })

        # Pagination: move to next week
        next_date = schedule.get("nextStartDate")
        if next_date is None or next_date <= current_date:
            break

        # Stop if we've clearly passed the end of the season
        # (playoffs end by late June at the latest)
        end_year = season + 1
        cutoff = f"{end_year}-07-01"
        if next_date > cutoff:
            break

        current_date = next_date

    game_ids = sorted(seen_ids)
    logger.info("Discovered %d games for season %d-%d", len(game_ids), season, season + 1)
    return game_metadata, game_ids


def save_game_metadata(metadata: list[dict], season: int) -> Path:
    """Write game-level schedule metadata to Parquet.

    Output: ``data/raw/schedule/season={season_code}/games.parquet``
    """
    if not metadata:
        logger.warning("No metadata to save for season %d", season)
        return Path()

    season_code = int(f"{season}{season + 1}")
    out_dir = RAW_SCHEDULE_DIR / f"season={season_code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "games.parquet"

    df = pd.DataFrame(metadata)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="snappy")

    logger.info("Saved schedule metadata → %s (%d games)", out_path, len(metadata))
    return out_path


# ---------------------------------------------------------------------------
# PBP flattening
# ---------------------------------------------------------------------------


def flatten_pbp(raw: dict) -> pd.DataFrame:
    """Flatten a single game's PBP JSON into a tabular DataFrame.

    Each row represents one play event.  Game-level context (teams, season,
    venue) is denormalized onto every row for query convenience.

    Parameters
    ----------
    raw : dict
        The full JSON response from the play-by-play endpoint.

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame with one row per event.
    """
    game_id = raw["id"]
    season = raw.get("season")
    game_type = raw.get("gameType")
    venue = raw.get("venue", {}).get("default", "")
    home_team = raw.get("homeTeam", {})
    away_team = raw.get("awayTeam", {})

    rows: list[dict] = []

    for play in raw.get("plays", []):
        details = play.get("details", {})
        period_desc = play.get("periodDescriptor", {})

        row = {
            # Game context
            "game_id": game_id,
            "season": season,
            "game_type": game_type,
            "venue": venue,
            "home_team_id": home_team.get("id"),
            "away_team_id": away_team.get("id"),
            "home_team_abbrev": home_team.get("abbrev", ""),
            "away_team_abbrev": away_team.get("abbrev", ""),
            # Event identifiers
            "event_id": play.get("eventId"),
            "period": period_desc.get("number"),
            "period_type": period_desc.get("periodType", ""),
            "time_in_period": play.get("timeInPeriod", ""),
            "time_remaining": play.get("timeRemaining", ""),
            "situation_code": play.get("situationCode", ""),
            "event_type": play.get("typeDescKey", ""),
            "event_type_code": play.get("typeCode"),
            # Coordinates
            "x_coord": details.get("xCoord"),
            "y_coord": details.get("yCoord"),
            # Key player IDs (nullable — not all events have these)
            "shooting_player_id": details.get("shootingPlayerId"),
            "scoring_player_id": details.get("scoringPlayerId"),
            "assist1_player_id": details.get("assist1PlayerId"),
            "assist2_player_id": details.get("assist2PlayerId"),
            "goalie_in_net_id": details.get("goalieInNetId"),
            # Shot specifics
            "shot_type": details.get("shotType"),
            "event_owner_team_id": details.get("eventOwnerTeamId"),
            "zone_code": details.get("zoneCode", ""),
            # Preserve all details as JSON for future feature engineering
            "details_json": json.dumps(details) if details else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("Game %d has no plays", game_id)
        return df

    # Enforce dtypes — nullable integers for IDs
    int_cols_nullable = [
        "shooting_player_id",
        "scoring_player_id",
        "assist1_player_id",
        "assist2_player_id",
        "goalie_in_net_id",
        "event_owner_team_id",
    ]
    for col in int_cols_nullable:
        if col in df.columns:
            df[col] = pd.array(df[col], dtype=pd.Int64Dtype())

    return df


def save_pbp_parquet(df: pd.DataFrame, game_id: int, season_code: int) -> Path:
    """Write a flattened PBP DataFrame to a partitioned Parquet file.

    Output: ``data/raw/pbp/season={season_code}/game_{game_id}.parquet``
    """
    out_dir = RAW_PBP_DIR / f"season={season_code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"game_{game_id}.parquet"

    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="snappy")

    logger.debug("Saved PBP → %s (%d events)", out_path, len(df))
    return out_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def scrape_season(
    client: NHLAPIClient,
    season: int,
    *,
    resume: bool = True,
    max_games: int | None = None,
) -> dict:
    """Scrape all PBP data for a single season.

    Parameters
    ----------
    client : NHLAPIClient
        Shared API client.
    season : int
        Start year of the season (e.g. 2024).
    resume : bool
        If True, skip games already in the checkpoint.
    max_games : int | None
        If set, stop after scraping this many games (useful for testing).

    Returns
    -------
    dict
        Summary with keys ``scraped``, ``skipped``, ``failed``, ``total``.
    """
    season_code = int(f"{season}{season + 1}")

    # Discover games
    metadata, game_ids = discover_game_ids(client, season)
    save_game_metadata(metadata, season)

    # Checkpoint
    ckpt = CheckpointManager("pbp", season) if resume else None
    skipped = 0
    scraped = 0
    failed = 0

    for game_id in game_ids:
        if max_games is not None and scraped >= max_games:
            logger.info("Reached --max-games=%d, stopping", max_games)
            break

        if ckpt and ckpt.is_complete(game_id):
            skipped += 1
            continue

        try:
            raw = client.get_pbp(game_id)
            df = flatten_pbp(raw)
            if not df.empty:
                save_pbp_parquet(df, game_id, season_code)
            if ckpt:
                ckpt.mark_complete(game_id)
            scraped += 1
            if scraped % 50 == 0:
                logger.info("Progress: %d/%d games scraped", scraped, len(game_ids))
        except Exception as exc:
            logger.error("Failed to scrape game %d: %s", game_id, exc)
            if ckpt:
                ckpt.mark_failed(game_id, str(exc))
            failed += 1

    summary = {
        "season": season,
        "total": len(game_ids),
        "scraped": scraped,
        "skipped": skipped,
        "failed": failed,
    }
    logger.info(
        "Season %d-%d complete: %d scraped, %d skipped, %d failed (of %d total)",
        season,
        season + 1,
        scraped,
        skipped,
        failed,
        len(game_ids),
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Scrape NHL play-by-play data and store as Parquet.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Season start years to scrape (e.g. 2024 for the 2024-25 season).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and re-scrape everything.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Stop after scraping N games per season (for testing).",
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Scrape all seasons from 2007-08 through 2025-26.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args(argv)

    if not args.seasons and not args.all_seasons:
        parser.error("You must provide either --seasons or --all-seasons.")

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    seasons = ALL_SEASONS if args.all_seasons else args.seasons
    resume = not args.no_resume

    with NHLAPIClient() as client:
        for season in seasons:
            scrape_season(
                client,
                season,
                resume=resume,
                max_games=args.max_games,
            )


if __name__ == "__main__":
    main()
