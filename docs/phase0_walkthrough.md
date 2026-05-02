# Phase 0: Data Ingestion Pipeline Walkthrough

I have successfully implemented the complete Data Ingestion Pipeline exactly as specified in the Phase 0 master plan. 

## 1. What was built

We built a production-grade scraping and validation system:

*   **API Client (`src/ingestion/api_client.py`)**: A centralized `requests` client with strict 1 req/sec rate limiting and exponential backoff for resilience against NHL server errors.
*   **Checkpointing (`src/ingestion/checkpoint.py`)**: A JSON-backed, crash-safe checkpointing system allowing interrupted scrapes to resume exactly where they left off.
*   **Play-By-Play Scraper (`src/ingestion/scrape_pbp.py`)**: Discovers games from the schedule API, downloads event data, flattens deeply nested JSON into tabular formats, preserves the raw `details_json`, and saves directly to partitioned Parquet files (`data/raw/pbp/season=...`).
*   **Shift Scraper (`src/ingestion/scrape_shifts.py`)**: Scrapes the separate shift-chart endpoint to track player ice-times, using the same game-id discovery and Parquet writing logic.
*   **DuckDB Schema (`src/ingestion/schema.py`)**: Loads the raw Parquet partitions into high-performance columnar DuckDB tables and creates a `v_shots` view for downstream feature engineering.
*   **Validators (`src/ingestion/validators.py`)**: An extensive data quality layer enforcing coordinate bounds, detecting arena tracking biases (>2σ deviation), flagging impossible time sequences, and auditing missing data.

## 2. Testing and Validation

I created a virtual environment with Python 3.12 and installed all core dependencies (`duckdb`, `pyarrow`, `pandas`, `requests`, `pytest`).

### Unit Tests
I wrote a comprehensive test suite (`tests/test_ingestion.py`) covering all logic, including mocked API responses to verify the rate-limiting and backoff behaviors. 

**Result**: `25/25 tests passed`.

### End-to-End Smoke Test
To ensure the pipeline works against the live NHL API, I ran a smoke test for a single game from the 2024-2025 season:

```bash
# 1. Scrape 1 game of PBP
> python -m src.ingestion.scrape_pbp --seasons 2024 --max-games 1
[INFO] Discovered 1398 games for season 2024-2025
[INFO] Saved PBP → data\raw\pbp\season=20242025\game_2024020001.parquet (349 events)

# 2. Scrape 1 game of shifts
> python -m src.ingestion.scrape_shifts --seasons 2024 --max-games 1
[INFO] Saved shifts → data\raw\shifts\season=20242025\game_2024020001.parquet (821 shifts)

# 3. Load DuckDB Schema
> python -m src.ingestion.schema
[INFO] Created table raw_pbp         — 349 rows
[INFO] Created table raw_shifts      — 821 rows
[INFO] Created table game_schedule   — 1398 rows
[INFO] Created view  v_shots          — 90 rows

# 4. Run Validators
> python -m src.ingestion.validators
[PASS] Coordinate Bounds
   0 out-of-bounds coordinates found (0.000% of 90 shots with coordinates)
[PASS] Arena Bias Detection
   0 arena(s) flagged for bias (>2.0 sigma from league mean) out of 1 total arenas
[PASS] Event Sequence Validation
   0 temporal anomalies found, 0 games with >5 anomalies
[PASS] Missing Data Audit
   Max null-coordinate rate by season: 0.0%. 0 season/arena combinations exceed 10.0% threshold.

Result: ALL CHECKS PASSED
```

## 3. Next Steps
Phase 0 is complete. The system is ready to ingest the full 19-year dataset whenever you're ready (by running `python -m src.ingestion.scrape_pbp --all-seasons`). 

We can now move on to **Phase 1: Feature Engineering & Baseline xG Model**.
