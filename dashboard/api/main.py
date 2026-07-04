"""
GSAx 2.0 Analytics API
Serves pre-computed goaltender metrics from Parquet files via DuckDB.
No API keys required — everything is self-contained.
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, FileResponse
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Optional

app = FastAPI(title="GSAx 2.0 Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Paths (absolute, script-relative) ----------
UI_DIR = Path(__file__).parent.parent / "ui"
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

SCORED_PATH   = DATA_DIR / "scored_shots.parquet"
PROJ_PATH     = DATA_DIR / "kalman_projections.parquet"
MAP_PATH      = DATA_DIR / "goalie_map.parquet"
DSIS_PATH     = DATA_DIR / "dsis_posteriors.parquet"
TEAM_DSIS_PATH= DATA_DIR / "dsis_team_effects.parquet"
BASE_PATH     = DATA_DIR / "goalie_base_metrics.parquet"

# Mount static UI files
app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


# ---------- Helpers ----------
def _json(df: pd.DataFrame) -> Response:
    """Return a DataFrame as a JSON response, handling NaN → null."""
    return Response(
        content=df.to_json(orient="records", default_handler=str),
        media_type="application/json",
    )


def _conn():
    """Create a fresh in-memory DuckDB connection with scored_shots registered."""
    c = duckdb.connect(":memory:")
    c.execute(f"""
        CREATE VIEW scored AS
        SELECT * FROM read_parquet('{str(SCORED_PATH).replace(chr(92), '/')}')
    """)
    return c


# ---------- Routes ----------

@app.get("/")
def root():
    return FileResponse(str(UI_DIR / "index.html"))


@app.get("/api/meta")
def get_metadata():
    """Return available seasons and teams for filter dropdowns.
    No API key needed — everything is derived from the local data."""
    c = _conn()
    seasons = c.execute("SELECT DISTINCT season FROM scored ORDER BY season DESC").df()
    teams = c.execute("""
        SELECT DISTINCT team FROM (
            SELECT home_team_abbrev AS team FROM scored
            UNION
            SELECT away_team_abbrev AS team FROM scored
        ) t
        WHERE team IS NOT NULL
        ORDER BY team
    """).df()
    return {
        "seasons": seasons["season"].tolist(),
        "teams": teams["team"].tolist(),
    }


@app.get("/api/leaderboard")
def get_leaderboard(
    season: Optional[int] = Query(None, description="e.g. 20252026"),
    team: Optional[str] = Query(None, description="e.g. TOR"),
    min_shots: int = Query(200, description="Minimum shots faced"),
):
    """Dynamic leaderboard with full filter support."""
    c = _conn()

    # Load name map and DSIS posteriors
    name_map = pd.read_parquet(MAP_PATH)
    dsis = pd.read_parquet(DSIS_PATH) if DSIS_PATH.exists() else pd.DataFrame()

    where = ["1=1"]
    if season:
        where.append(f"season = {season}")
    if team:
        # goalie's team: if event_owner is home, goalie is away, vice versa
        where.append(f"""(
            (event_owner_team_id = home_team_id AND away_team_abbrev = '{team}')
            OR
            (event_owner_team_id = away_team_id AND home_team_abbrev = '{team}')
        )""")

    where_clause = " AND ".join(where)

    df = c.execute(f"""
        SELECT
            goalie_in_net_id AS goalie_id,
            COUNT(*)         AS shots,
            SUM(is_goal)     AS goals,
            1.0 - (CAST(SUM(is_goal) AS DOUBLE) / COUNT(*)) AS sv_pct,
            SUM(xg_prob) - SUM(is_goal)                     AS gsax,
            AVG(traffic_density)                             AS avg_traffic,
            AVG(puck_speed)                                  AS avg_speed,
            AVG(ABS(delta_angle))                            AS avg_movement
        FROM scored
        WHERE {where_clause}
        GROUP BY 1
        HAVING COUNT(*) >= {min_shots}
        ORDER BY gsax DESC
    """).df()

    # Merge names
    df = df.merge(name_map, on="goalie_id", how="left")
    df["goalie_name"] = df["goalie_name"].fillna(df["goalie_id"].astype(str))

    # Merge DSIS talent if available
    if not dsis.empty and "goalie_id" in dsis.columns:
        talent_col = [c for c in dsis.columns if "talent" in c.lower() and "per_shot" in c.lower()]
        if talent_col:
            df = df.merge(dsis[["goalie_id", talent_col[0]]].rename(
                columns={talent_col[0]: "isolated_talent"}
            ), on="goalie_id", how="left")
            
    # Merge RCI if available
    if BASE_PATH.exists():
        base_df = pd.read_parquet(BASE_PATH)
        # Get the latest season RCI for each goalie
        if season:
            b_df = base_df[base_df['season'] == int(season)]
        else:
            b_df = base_df.sort_values('season').drop_duplicates('goalie_in_net_id', keep='last')
        df = df.merge(b_df[['goalie_in_net_id', 'rci']].rename(columns={'goalie_in_net_id': 'goalie_id'}), on='goalie_id', how='left')

    # Percentiles (within this filtered set)
    if len(df) > 0:
        df["gsax_pct"]    = df["gsax"].rank(pct=True) * 100
        df["traffic_pct"] = df["avg_traffic"].rank(pct=True) * 100
        df["speed_pct"]   = df["avg_speed"].rank(pct=True) * 100
        df["movement_pct"]= df["avg_movement"].rank(pct=True) * 100
        if 'rci' in df.columns:
            df["rci_pct"] = df["rci"].rank(pct=True) * 100

    return _json(df)


@app.get("/api/goalie/{goalie_id}")
def get_goalie_detail(goalie_id: int):
    """Full career breakdown for a single goalie."""
    c = _conn()
    name_map = pd.read_parquet(MAP_PATH)

    # Career stats by season
    career = c.execute(f"""
        SELECT
            season,
            COUNT(*)     AS shots,
            SUM(is_goal) AS goals,
            1.0 - (CAST(SUM(is_goal) AS DOUBLE) / COUNT(*)) AS sv_pct,
            SUM(xg_prob) - SUM(is_goal)                     AS gsax,
            AVG(traffic_density)                             AS avg_traffic,
            AVG(puck_speed)                                  AS avg_speed
        FROM scored
        WHERE goalie_in_net_id = {goalie_id}
        GROUP BY 1
        ORDER BY 1
    """).df()

    # Recent shots for the shot map
    shots = c.execute(f"""
        SELECT adjusted_x, adjusted_y, xg_prob, is_goal,
               traffic_density, puck_speed, shot_type, shot_distance
        FROM scored
        WHERE goalie_in_net_id = {goalie_id}
        ORDER BY game_id DESC, event_id DESC
        LIMIT 500
    """).df()

    # Name
    row = name_map[name_map["goalie_id"] == goalie_id]
    name = row["goalie_name"].iloc[0] if len(row) else str(goalie_id)

    # Projections
    proj_data = {}
    if PROJ_PATH.exists():
        proj = pd.read_parquet(PROJ_PATH)
        p = proj[proj["goalie_id"] == goalie_id]
        if len(p):
            proj_data = p.iloc[0].to_dict()

    # Base metrics (RCI, etc.)
    base_data = []
    if BASE_PATH.exists():
        b_df = pd.read_parquet(BASE_PATH)
        b_df = b_df[b_df["goalie_in_net_id"] == goalie_id]
        base_data = b_df.where(pd.notnull(b_df), None).to_dict(orient="records")

    result = {
        "goalie_id": goalie_id,
        "name": name,
        "career": career.where(pd.notnull(career), None).to_dict(orient="records"),
        "recent_shots": shots.where(pd.notnull(shots), None).to_dict(orient="records"),
        "base_metrics": base_data,
        "projection": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in proj_data.items()},
    }
    # Serialize with pandas to handle any remaining NaN without flattening nested dicts
    return Response(
        content=pd.DataFrame([result]).to_json(orient="records")[1:-1],
        media_type="application/json",
    )


@app.get("/api/projections")
def get_projections(
    min_shots: int = Query(0, description="Filter goalies who have faced at least this many career shots (to filter out extreme noise)"),
):
    """Returns true talent projections for the upcoming season.
    Only includes goalies who appeared in the most recent season."""
    if not PROJ_PATH.exists():
        return Response(content="[]", media_type="application/json")
        
    proj = pd.read_parquet(PROJ_PATH)
    name_map = pd.read_parquet(MAP_PATH)
    
    # Only keep goalies who played in the most recent season
    c = _conn()
    max_season = c.execute("SELECT MAX(season) FROM scored").fetchone()[0]
    active_ids = c.execute(f"""
        SELECT DISTINCT goalie_in_net_id AS goalie_id
        FROM scored
        WHERE season = {max_season}
    """).df()
    proj = proj[proj["goalie_id"].isin(active_ids["goalie_id"])]
    
    # Merge names
    df = proj.merge(name_map, on="goalie_id", how="left")
    df["goalie_name"] = df["goalie_name"].fillna(df["goalie_id"].astype(str))
    
    return Response(
        content=df.to_json(orient="records", default_handler=str),
        media_type="application/json"
    )

@app.get("/api/team-defense")
def get_team_defense(season: Optional[int] = Query(None)):
    """Returns team defense impact scores (DSIS)."""
    if not TEAM_DSIS_PATH.exists():
        return Response(content="[]", media_type="application/json")
        
    df = pd.read_parquet(TEAM_DSIS_PATH)
    if season:
        df = df[df['season'] == int(season)]
        
    # Get team names mapping from scored_shots (hacky but works without a dedicated table)
    c = _conn()
    teams = c.execute("SELECT DISTINCT home_team_id as team_id, home_team_abbrev as team_name FROM scored WHERE home_team_abbrev IS NOT NULL").df()
    df = df.merge(teams, on='team_id', how='left')
    df['team_name'] = df['team_name'].fillna(df['team_id'].astype(str))
    
    # Sort by defense impact (negative is good: taking away xG)
    df = df.sort_values('dsis_team_defense_impact_per_shot', ascending=True)
    
    return Response(
        content=df.to_json(orient="records", default_handler=str),
        media_type="application/json"
    )

@app.get("/api/head2head")
def head_to_head(
    g1: int = Query(..., description="Goalie 1 ID"),
    g2: int = Query(..., description="Goalie 2 ID"),
    season: Optional[int] = Query(None),
):
    """Compare two goalies side by side."""
    c = _conn()
    name_map = pd.read_parquet(MAP_PATH)

    where = ""
    if season:
        where = f"AND season = {season}"

    rows = []
    for gid in [g1, g2]:
        r = c.execute(f"""
            SELECT
                {gid}            AS goalie_id,
                COUNT(*)         AS shots,
                SUM(is_goal)     AS goals,
                1.0 - (CAST(SUM(is_goal) AS DOUBLE) / COUNT(*)) AS sv_pct,
                SUM(xg_prob) - SUM(is_goal)                     AS gsax,
                AVG(traffic_density)                             AS avg_traffic,
                AVG(puck_speed)                                  AS avg_speed,
                AVG(ABS(delta_angle))                            AS avg_movement
            FROM scored
            WHERE goalie_in_net_id = {gid} {where}
        """).df()
        rows.append(r)

    df = pd.concat(rows, ignore_index=True)
    df = df.merge(name_map, on="goalie_id", how="left")
    df["goalie_name"] = df["goalie_name"].fillna(df["goalie_id"].astype(str))
    return _json(df)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
