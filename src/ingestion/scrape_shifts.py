"""
NHL API shift chart scraper.

Pulls player shift data (start/end times per shift per player per game)
from the NHL stats API. This data is essential for:
- Determining which players are on the ice at the time of each shot
- Calculating goalie shift duration at the moment of each shot
- Estimating screening (counting skaters in shooting lanes)

Endpoint:
    https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}

Usage:
    python -m src.ingestion.scrape_shifts --seasons 2024 2025
    python -m src.ingestion.scrape_shifts --seasons 2024 --resume --max-games 5
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.api_client import NHLAPIClient
from src.ingestion.checkpoint import CheckpointManager
from src.ingestion.scrape_pbp import ALL_SEASONS, discover_game_ids

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_SHIFTS_DIR = Path("data/raw/shifts")


# ---------------------------------------------------------------------------
# Shift flattening
# ---------------------------------------------------------------------------


def flatten_shifts(raw: dict, game_id: int) -> pd.DataFrame:
    """Flatten a single game's shift chart JSON into a tabular DataFrame.

    Each row represents one player shift.

    Parameters
    ----------
    raw : dict
        The full JSON response from the shift chart endpoint.
    game_id : int
        The game ID (passed explicitly since the response structure
        uses ``gameId`` inside each shift record).

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame with one row per shift.
    """
    shifts = raw.get("data", [])
    if not shifts:
        logger.warning("Game %d has no shift data", game_id)
        return pd.DataFrame()

    rows: list[dict] = []
    for shift in shifts:
        rows.append({
            "game_id": shift.get("gameId", game_id),
            "player_id": shift.get("playerId"),
            "team_id": shift.get("teamId"),
            "team_abbrev": shift.get("teamAbbrev", ""),
            "first_name": shift.get("firstName", ""),
            "last_name": shift.get("lastName", ""),
            "period": shift.get("period"),
            "start_time": shift.get("startTime", ""),
            "end_time": shift.get("endTime", ""),
            "duration": shift.get("duration", ""),
            "shift_number": shift.get("shiftNumber"),
            "type_code": shift.get("typeCode"),
        })

    df = pd.DataFrame(rows)

    # Enforce dtypes
    int_cols_nullable = ["player_id", "team_id", "shift_number", "type_code"]
    for col in int_cols_nullable:
        if col in df.columns:
            df[col] = pd.array(df[col], dtype=pd.Int64Dtype())

    if "period" in df.columns:
        df["period"] = pd.array(df["period"], dtype=pd.Int8Dtype())

    return df


def save_shifts_parquet(df: pd.DataFrame, game_id: int, season_code: int) -> Path:
    """Write a flattened shifts DataFrame to a partitioned Parquet file.

    Output: ``data/raw/shifts/season={season_code}/game_{game_id}.parquet``
    """
    out_dir = RAW_SHIFTS_DIR / f"season={season_code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"game_{game_id}.parquet"

    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="snappy")

    logger.debug("Saved shifts → %s (%d shifts)", out_path, len(df))
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
    """Scrape all shift chart data for a single season.

    Uses the same game-ID discovery as the PBP scraper. If PBP has already
    been scraped for this season, the schedule metadata cache is used
    automatically (the API call is still made to get the game list, but
    it's lightweight).

    Parameters
    ----------
    client : NHLAPIClient
        Shared API client.
    season : int
        Start year of the season (e.g. 2024).
    resume : bool
        If True, skip games already in the checkpoint.
    max_games : int | None
        Stop after scraping this many games (for testing).

    Returns
    -------
    dict
        Summary with keys ``scraped``, ``skipped``, ``failed``, ``total``.
    """
    season_code = int(f"{season}{season + 1}")

    # Discover games (reuse PBP's discovery logic)
    _metadata, game_ids = discover_game_ids(client, season)

    # Checkpoint
    ckpt = CheckpointManager("shifts", season) if resume else None
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
            raw = client.get_shifts(game_id)
            df = flatten_shifts(raw, game_id)
            if not df.empty:
                save_shifts_parquet(df, game_id, season_code)
            if ckpt:
                ckpt.mark_complete(game_id)
            scraped += 1
            if scraped % 50 == 0:
                logger.info("Progress: %d/%d games scraped", scraped, len(game_ids))
        except Exception as exc:
            logger.error("Failed to scrape shifts for game %d: %s", game_id, exc)
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
        "Shifts season %d-%d: %d scraped, %d skipped, %d failed (of %d total)",
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
        description="Scrape NHL shift chart data and store as Parquet.",
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
