"""
Data ingestion module.

Handles scraping and caching of NHL API data:
- Play-by-play event data with x,y coordinates
- Shift chart data (player on-ice intervals)
- NHL EDGE tracking stats (skating speed, distance, acceleration)

Sub-modules:
    api_client   — Shared rate-limited HTTP client
    checkpoint   — Checkpoint/resume system for incremental scraping
    scrape_pbp   — Play-by-play scraper
    scrape_shifts— Shift chart scraper
    scrape_edge  — NHL EDGE tracking data scraper (stub)
    schema       — DuckDB schema definition & Parquet loader
    validators   — Data quality validation layer
"""

from src.ingestion.api_client import NHLAPIClient
from src.ingestion.checkpoint import CheckpointManager
from src.ingestion.schema import init_database, load_parquet_tables

__all__ = [
    "NHLAPIClient",
    "CheckpointManager",
    "init_database",
    "load_parquet_tables",
]
