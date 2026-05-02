"""
Data quality validation layer.

Implements the four validation checks specified in the master implementation
plan (§0.4):

1. **Coordinate sanity checks** — x ∈ [-100, 100], y ∈ [-42.5, 42.5]
2. **Arena bias detection** — per-arena shot location distributions, flag outliers
3. **Event sequence validation** — detect impossible event orderings
4. **Missing data audit** — null coordinate rates by season and arena

Usage:
    python -m src.ingestion.validators --season 2024
    python -m src.ingestion.validators --all-seasons
"""

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from src.ingestion.schema import DB_PATH, init_database, load_parquet_tables

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

X_MIN, X_MAX = -100.0, 100.0
Y_MIN, Y_MAX = -42.5, 42.5

# Arena bias: flag arenas whose mean shot location deviates by this many
# standard deviations from the league-wide mean
ARENA_BIAS_SIGMA_THRESHOLD = 2.0

# Missing data: flag season/arena combos with more than this percentage
# of null coordinates on shot events
MISSING_COORD_THRESHOLD_PCT = 10.0

# Sequence validation: minimum plausible seconds between certain event pairs
# (e.g. shot → faceoff → shot shouldn't happen in < 2 seconds)
MIN_PLAUSIBLE_SEQUENCE_SECONDS = 2.0

# Flag games with more than this many temporal anomalies
MAX_TEMPORAL_ANOMALIES_PER_GAME = 5

# Shot event types
SHOT_EVENT_TYPES = ("shot-on-goal", "goal", "missed-shot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _time_to_seconds(time_str: str) -> float | None:
    """Convert ``MM:SS`` to total seconds.  Returns None on parse failure."""
    if not time_str:
        return None
    match = re.match(r"^(\d+):(\d{2})$", time_str.strip())
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    summary: str
    details: pd.DataFrame | None = None


@dataclass
class ValidationReport:
    """Aggregate report from all validation checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print("\n" + "=" * 60)
        print("  DATA QUALITY VALIDATION REPORT")
        print("=" * 60 + "\n")

        for check in self.checks:
            icon = "[PASS]" if check.passed else "[FAIL]"
            print(f"  {icon} {check.name}")
            print(f"     {check.summary}")
            if check.details is not None and not check.details.empty:
                # Print first few rows of details
                detail_str = check.details.head(10).to_string(index=False)
                for line in detail_str.split("\n"):
                    print(f"       {line}")
                if len(check.details) > 10:
                    print(f"       ... and {len(check.details) - 10} more rows")
            print()

        status = "ALL CHECKS PASSED" if self.all_passed else "SOME CHECKS FAILED"
        print(f"  Result: {status}\n")


# ---------------------------------------------------------------------------
# 1. Coordinate sanity checks
# ---------------------------------------------------------------------------


def check_coordinate_bounds(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    """Verify that shot coordinates fall within NHL rink dimensions.

    Checks: x ∈ [-100, 100], y ∈ [-42.5, 42.5] for all shot events.
    Logs violations but does NOT clip values.
    """
    sql = f"""
        SELECT
            game_id,
            event_id,
            x_coord,
            y_coord,
            event_type,
            venue
        FROM raw_pbp
        WHERE event_type IN {SHOT_EVENT_TYPES}
          AND x_coord IS NOT NULL
          AND y_coord IS NOT NULL
          AND (
              x_coord < {X_MIN} OR x_coord > {X_MAX}
              OR y_coord < {Y_MIN} OR y_coord > {Y_MAX}
          )
        ORDER BY game_id, event_id
    """
    violations = conn.execute(sql).fetchdf()

    total_shots = conn.execute(f"""
        SELECT COUNT(*) FROM raw_pbp
        WHERE event_type IN {SHOT_EVENT_TYPES}
          AND x_coord IS NOT NULL AND y_coord IS NOT NULL
    """).fetchone()[0]

    n_violations = len(violations)
    pct = (n_violations / total_shots * 100) if total_shots > 0 else 0

    passed = n_violations == 0
    summary = (
        f"{n_violations:,} out-of-bounds coordinates found "
        f"({pct:.3f}% of {total_shots:,} shots with coordinates)"
    )

    if not passed:
        logger.warning("Coordinate bounds check FAILED: %s", summary)

    return CheckResult(
        name="Coordinate Bounds",
        passed=passed,
        summary=summary,
        details=violations if not passed else None,
    )


# ---------------------------------------------------------------------------
# 2. Arena bias detection
# ---------------------------------------------------------------------------


def check_arena_bias(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    """Detect arenas with anomalous shot location distributions.

    Computes per-arena mean and std of shot x/y coordinates, then flags
    arenas whose means deviate more than 2σ from the league-wide mean.
    """
    sql = f"""
        WITH arena_stats AS (
            SELECT
                venue,
                COUNT(*) AS n_shots,
                AVG(x_coord) AS mean_x,
                AVG(y_coord) AS mean_y,
                STDDEV_SAMP(x_coord) AS std_x,
                STDDEV_SAMP(y_coord) AS std_y
            FROM raw_pbp
            WHERE event_type IN {SHOT_EVENT_TYPES}
              AND x_coord IS NOT NULL
              AND y_coord IS NOT NULL
              AND venue IS NOT NULL
              AND venue != ''
            GROUP BY venue
            HAVING COUNT(*) >= 100
        ),
        league_stats AS (
            SELECT
                AVG(mean_x) AS league_mean_x,
                AVG(mean_y) AS league_mean_y,
                STDDEV_SAMP(mean_x) AS league_std_x,
                STDDEV_SAMP(mean_y) AS league_std_y
            FROM arena_stats
        )
        SELECT
            a.venue,
            a.n_shots,
            ROUND(a.mean_x, 2) AS mean_x,
            ROUND(a.mean_y, 2) AS mean_y,
            ROUND(a.std_x, 2) AS std_x,
            ROUND(a.std_y, 2) AS std_y,
            ROUND(ABS(a.mean_x - l.league_mean_x) / NULLIF(l.league_std_x, 0), 2)
                AS x_zscore,
            ROUND(ABS(a.mean_y - l.league_mean_y) / NULLIF(l.league_std_y, 0), 2)
                AS y_zscore
        FROM arena_stats a
        CROSS JOIN league_stats l
        WHERE ABS(a.mean_x - l.league_mean_x) / NULLIF(l.league_std_x, 0)
                > {ARENA_BIAS_SIGMA_THRESHOLD}
           OR ABS(a.mean_y - l.league_mean_y) / NULLIF(l.league_std_y, 0)
                > {ARENA_BIAS_SIGMA_THRESHOLD}
        ORDER BY x_zscore DESC
    """
    flagged = conn.execute(sql).fetchdf()
    n_flagged = len(flagged)

    total_arenas = conn.execute(f"""
        SELECT COUNT(DISTINCT venue) FROM raw_pbp
        WHERE event_type IN {SHOT_EVENT_TYPES}
          AND venue IS NOT NULL AND venue != ''
    """).fetchone()[0]

    passed = n_flagged == 0
    summary = (
        f"{n_flagged} arena(s) flagged for bias "
        f"(>{ARENA_BIAS_SIGMA_THRESHOLD} sigma from league mean) "
        f"out of {total_arenas} total arenas"
    )

    if not passed:
        for _, row in flagged.iterrows():
            logger.warning(
                "Arena bias: %s — mean_x=%.2f, mean_y=%.2f, x_z=%.2f, y_z=%.2f",
                row["venue"],
                row["mean_x"],
                row["mean_y"],
                row["x_zscore"],
                row["y_zscore"],
            )

    return CheckResult(
        name="Arena Bias Detection",
        passed=passed,
        summary=summary,
        details=flagged if not passed else None,
    )


# ---------------------------------------------------------------------------
# 3. Event sequence validation
# ---------------------------------------------------------------------------


def check_event_sequences(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    """Detect impossible event sequences in play-by-play data.

    Flags cases where events happen impossibly fast (e.g. shot → faceoff →
    shot within 0.2 seconds) or where timestamps are not monotonically
    non-decreasing within a period.
    """
    # Fetch PBP ordered by game/period/time for analysis
    sql = """
        SELECT
            game_id,
            event_id,
            period,
            time_in_period,
            event_type
        FROM raw_pbp
        WHERE time_in_period IS NOT NULL
          AND time_in_period != ''
        ORDER BY game_id, period, time_in_period
    """
    df = conn.execute(sql).fetchdf()

    if df.empty:
        return CheckResult(
            name="Event Sequence Validation",
            passed=True,
            summary="No PBP data to validate",
        )

    # Convert times
    df["seconds"] = df["time_in_period"].apply(_time_to_seconds)
    df = df.dropna(subset=["seconds"])

    # Check for non-monotonic timestamps within each game/period
    anomaly_rows: list[dict] = []
    games_with_anomalies: set[int] = set()

    for (game_id, period), group in df.groupby(["game_id", "period"]):
        times = group["seconds"].values
        events = group["event_type"].values
        event_ids = group["event_id"].values

        game_anomalies = 0
        for i in range(1, len(times)):
            elapsed = times[i] - times[i - 1]

            # Check for backwards time (non-monotonic)
            if elapsed < 0:
                game_anomalies += 1
                anomaly_rows.append({
                    "game_id": game_id,
                    "period": period,
                    "event_id": int(event_ids[i]),
                    "prev_event": events[i - 1],
                    "curr_event": events[i],
                    "time_gap_s": float(elapsed),
                    "issue": "non_monotonic",
                })

            # Check for impossibly fast shot sequences
            if (
                elapsed >= 0
                and elapsed < MIN_PLAUSIBLE_SEQUENCE_SECONDS
                and events[i - 1] in ("faceoff", "stoppage")
                and events[i] in SHOT_EVENT_TYPES
                and i >= 2
                and events[i - 2] in SHOT_EVENT_TYPES
            ):
                game_anomalies += 1
                anomaly_rows.append({
                    "game_id": game_id,
                    "period": period,
                    "event_id": int(event_ids[i]),
                    "prev_event": events[i - 1],
                    "curr_event": events[i],
                    "time_gap_s": float(elapsed),
                    "issue": "impossible_sequence",
                })

        if game_anomalies > MAX_TEMPORAL_ANOMALIES_PER_GAME:
            games_with_anomalies.add(game_id)

    anomalies_df = pd.DataFrame(anomaly_rows) if anomaly_rows else pd.DataFrame()
    n_anomalies = len(anomaly_rows)
    n_flagged_games = len(games_with_anomalies)

    passed = n_flagged_games == 0
    summary = (
        f"{n_anomalies} temporal anomalies found, "
        f"{n_flagged_games} games with >{MAX_TEMPORAL_ANOMALIES_PER_GAME} anomalies"
    )

    return CheckResult(
        name="Event Sequence Validation",
        passed=passed,
        summary=summary,
        details=anomalies_df if not anomalies_df.empty else None,
    )


# ---------------------------------------------------------------------------
# 4. Missing data audit
# ---------------------------------------------------------------------------


def check_missing_data(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    """Audit null coordinate rates on shot events, by season and arena.

    Flags season/arena combinations where >10% of shots have null
    x_coord or y_coord.
    """
    # By season
    sql_season = f"""
        SELECT
            season,
            COUNT(*) AS total_shots,
            SUM(CASE WHEN x_coord IS NULL OR y_coord IS NULL THEN 1 ELSE 0 END)
                AS null_coords,
            ROUND(
                100.0 * SUM(CASE WHEN x_coord IS NULL OR y_coord IS NULL THEN 1 ELSE 0 END)
                / COUNT(*),
                2
            ) AS null_pct
        FROM raw_pbp
        WHERE event_type IN {SHOT_EVENT_TYPES}
        GROUP BY season
        ORDER BY season
    """
    by_season = conn.execute(sql_season).fetchdf()

    # By arena
    sql_arena = f"""
        SELECT
            season,
            venue,
            COUNT(*) AS total_shots,
            SUM(CASE WHEN x_coord IS NULL OR y_coord IS NULL THEN 1 ELSE 0 END)
                AS null_coords,
            ROUND(
                100.0 * SUM(CASE WHEN x_coord IS NULL OR y_coord IS NULL THEN 1 ELSE 0 END)
                / COUNT(*),
                2
            ) AS null_pct
        FROM raw_pbp
        WHERE event_type IN {SHOT_EVENT_TYPES}
          AND venue IS NOT NULL AND venue != ''
        GROUP BY season, venue
        HAVING (
            100.0 * SUM(CASE WHEN x_coord IS NULL OR y_coord IS NULL THEN 1 ELSE 0 END)
            / COUNT(*)
        ) > {MISSING_COORD_THRESHOLD_PCT}
        ORDER BY null_pct DESC
    """
    flagged_arenas = conn.execute(sql_arena).fetchdf()

    # Overall
    overall_null_pct = by_season["null_pct"].max() if not by_season.empty else 0
    n_flagged = len(flagged_arenas)

    passed = n_flagged == 0 and overall_null_pct <= MISSING_COORD_THRESHOLD_PCT
    summary = (
        f"Max null-coordinate rate by season: {overall_null_pct:.1f}%. "
        f"{n_flagged} season/arena combinations exceed {MISSING_COORD_THRESHOLD_PCT}% threshold."
    )

    # Combine details
    details_parts = []
    if not by_season.empty:
        by_season["scope"] = "season"
        details_parts.append(by_season)
    if not flagged_arenas.empty:
        flagged_arenas["scope"] = "season_arena"
        details_parts.append(flagged_arenas)

    details = pd.concat(details_parts, ignore_index=True) if details_parts else None

    return CheckResult(
        name="Missing Data Audit",
        passed=passed,
        summary=summary,
        details=details,
    )


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------


def run_all_checks(conn: duckdb.DuckDBPyConnection) -> ValidationReport:
    """Execute all four validation checks and return a combined report."""
    report = ValidationReport()

    # Check that raw_pbp exists
    tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    if "raw_pbp" not in tables:
        logger.error("raw_pbp table not found — run schema loader first")
        report.checks.append(CheckResult(
            name="Prerequisites",
            passed=False,
            summary="raw_pbp table not found in DuckDB. Run `python -m src.ingestion.schema` first.",
        ))
        return report

    logger.info("Running coordinate bounds check...")
    report.checks.append(check_coordinate_bounds(conn))

    logger.info("Running arena bias detection...")
    report.checks.append(check_arena_bias(conn))

    logger.info("Running event sequence validation...")
    report.checks.append(check_event_sequences(conn))

    logger.info("Running missing data audit...")
    report.checks.append(check_missing_data(conn))

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run data quality validation checks on ingested NHL data.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the DuckDB file (default: data/goaltender_analytics.duckdb).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload Parquet files into DuckDB before validating.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    conn = init_database(args.db_path)

    if args.reload:
        load_parquet_tables(conn)

    report = run_all_checks(conn)
    report.print_summary()

    conn.close()

    # Exit with non-zero status if any check failed
    if not report.all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
