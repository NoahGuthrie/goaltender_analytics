"""
DuckDB schema definition and Parquet loader.

Provides functions to:
- Initialize a DuckDB database at ``data/goaltender_analytics.duckdb``
- Load partitioned Parquet files into DuckDB tables via glob reads
- Create convenience views for downstream analysis (e.g. ``v_shots``)

Usage:
    python -m src.ingestion.schema          # load all Parquet → DuckDB
    python -m src.ingestion.schema --info    # print table row counts
"""

import argparse
import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("data/goaltender_analytics.duckdb")

RAW_PBP_GLOB = "data/raw/pbp/**/*.parquet"
RAW_SHIFTS_GLOB = "data/raw/shifts/**/*.parquet"
RAW_SCHEDULE_GLOB = "data/raw/schedule/**/*.parquet"

# Shot event types relevant for xG modeling
SHOT_EVENT_TYPES = ("shot-on-goal", "goal", "missed-shot")


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------


def init_database(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open (or create) a DuckDB database.

    Parameters
    ----------
    db_path : Path | str | None
        Path to the ``.duckdb`` file.  Defaults to ``data/goaltender_analytics.duckdb``.
        Pass ``":memory:"`` for an in-memory database (useful in tests).

    Returns
    -------
    duckdb.DuckDBPyConnection
    """
    path = str(db_path or DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(path)
    logger.info("Opened DuckDB database at %s", path)
    return conn


# ---------------------------------------------------------------------------
# Table creation from Parquet
# ---------------------------------------------------------------------------


def load_parquet_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """(Re)create DuckDB tables from partitioned Parquet directories.

    Tables created:
    - ``raw_pbp`` — all play-by-play events
    - ``raw_shifts`` — all shift chart records
    - ``game_schedule`` — game-level metadata (date, venue, teams)

    Views created:
    - ``v_shots`` — filtered to shot-relevant event types for xG modeling
    """
    _load_table(conn, "raw_pbp", RAW_PBP_GLOB)
    _load_table(conn, "raw_shifts", RAW_SHIFTS_GLOB)
    _load_table(conn, "game_schedule", RAW_SCHEDULE_GLOB)
    _create_views(conn)
    logger.info("All tables and views created successfully")


def _load_table(conn: duckdb.DuckDBPyConnection, table_name: str, glob_path: str) -> None:
    """Create or replace a table from a Parquet glob pattern."""
    # Check if any matching files exist
    from pathlib import Path as _P
    import glob as _glob

    matching = _glob.glob(glob_path, recursive=True)
    if not matching:
        logger.warning("No Parquet files found for %s at %s — skipping", table_name, glob_path)
        return

    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_parquet('{glob_path}', hive_partitioning=true)
    """
    conn.execute(sql)
    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info("Created table %-15s — %d rows", table_name, count)


def _create_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Create convenience views for downstream analysis."""
    # Check if raw_pbp exists before creating the view
    tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
    if "raw_pbp" not in tables:
        logger.warning("raw_pbp table not found — skipping v_shots view")
        return

    shot_types_sql = ", ".join(f"'{t}'" for t in SHOT_EVENT_TYPES)
    conn.execute(f"""
        CREATE OR REPLACE VIEW v_shots AS
        SELECT *
        FROM raw_pbp
        WHERE event_type IN ({shot_types_sql})
    """)

    count = conn.execute("SELECT COUNT(*) FROM v_shots").fetchone()[0]
    logger.info("Created view  v_shots          — %d rows", count)


# ---------------------------------------------------------------------------
# Info / diagnostics
# ---------------------------------------------------------------------------


def print_info(conn: duckdb.DuckDBPyConnection) -> None:
    """Print summary information about all tables in the database."""
    tables = conn.execute("SHOW TABLES").fetchall()
    print("\n=== DuckDB Table Summary ===\n")
    for (table_name,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name:<20s} {count:>12,d} rows")

    # Views
    # DuckDB doesn't have a simple SHOW VIEWS; query information_schema
    views = conn.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_type = 'VIEW'
    """).fetchall()
    if views:
        print()
        for (view_name,) in views:
            count = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
            print(f"  {view_name:<20s} {count:>12,d} rows (view)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Load Parquet files into DuckDB and create analytical views.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print table row counts instead of loading data.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override the DuckDB file path.",
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

    if args.info:
        print_info(conn)
    else:
        load_parquet_tables(conn)
        print_info(conn)

    conn.close()


if __name__ == "__main__":
    main()
