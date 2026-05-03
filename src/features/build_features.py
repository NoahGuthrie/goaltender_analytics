import duckdb
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def build_xg_features(db_path="data/goaltender_analytics.duckdb", output_path="data/processed/xg_features.parquet"):
    conn = duckdb.connect(db_path)
    
    logger.info("Computing base features and window functions via DuckDB...")
    
    # Base table with window functions, true attacking net resolution, and strength state
    query = """
    WITH team_nets AS (
        SELECT 
            game_id, period, event_owner_team_id,
            CASE WHEN avg(x_coord) > 0 THEN 89 ELSE -89 END as net_x
        FROM raw_pbp
        WHERE event_type IN ('shot-on-goal', 'goal', 'missed-shot') AND x_coord IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    pbp_with_prev AS (
        SELECT 
            p.game_id, p.season, p.game_type, p.venue, 
            p.home_team_id, p.away_team_id, p.home_team_abbrev, p.away_team_abbrev,
            p.event_id, p.period, p.period_type, p.time_in_period, p.situation_code,
            p.event_type, p.x_coord, p.y_coord,
            p.shooting_player_id, p.scoring_player_id, p.goalie_in_net_id,
            p.shot_type, p.event_owner_team_id, p.details_json,
            n.net_x,
            -- Time logic
            CAST(SPLIT_PART(p.time_in_period, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(p.time_in_period, ':', 2) AS INT) as time_seconds,
            
            -- Previous event logic
            LAG(p.event_type, 1) OVER w as prev_event_type,
            LAG(p.x_coord, 1) OVER w as prev_x,
            LAG(p.y_coord, 1) OVER w as prev_y,
            LAG(CAST(SPLIT_PART(p.time_in_period, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(p.time_in_period, ':', 2) AS INT), 1) OVER w as prev_time_seconds,
            LAG(p.event_type, 2) OVER w || '->' || LAG(p.event_type, 1) OVER w as sequence_2_events,
            
            -- Time since last stoppage (fatigue metric)
            COALESCE(
                MAX(CASE WHEN p.event_type = 'faceoff' THEN CAST(SPLIT_PART(p.time_in_period, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(p.time_in_period, ':', 2) AS INT) ELSE NULL END) 
                OVER (PARTITION BY p.game_id, p.period ORDER BY p.event_id ROWS UNBOUNDED PRECEDING),
                0 -- Default to start of period if no faceoff recorded
            ) as last_stoppage_time,
            
            -- Score tracking for score differential
            COALESCE(SUM(CASE WHEN p.event_type = 'goal' AND p.event_owner_team_id = p.home_team_id THEN 1 ELSE 0 END) 
                OVER (PARTITION BY p.game_id ORDER BY p.event_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) as home_goals,
            COALESCE(SUM(CASE WHEN p.event_type = 'goal' AND p.event_owner_team_id = p.away_team_id THEN 1 ELSE 0 END) 
                OVER (PARTITION BY p.game_id ORDER BY p.event_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) as away_goals,
            
            COUNT(CASE WHEN p.event_type IN ('shot-on-goal', 'goal') THEN 1 END) 
                OVER (PARTITION BY p.game_id ORDER BY CAST(SPLIT_PART(p.time_in_period, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(p.time_in_period, ':', 2) AS INT) RANGE BETWEEN 4 PRECEDING AND 1 PRECEDING) as shot_sequence_num,
            
            COUNT(CASE WHEN SIGN(p.x_coord) = SIGN(n.net_x) AND p.event_type IN ('hit', 'blocked-shot', 'faceoff', 'giveaway', 'takeaway', 'shot-on-goal', 'missed-shot') THEN 1 END)
                OVER (PARTITION BY p.game_id ORDER BY CAST(SPLIT_PART(p.time_in_period, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(p.time_in_period, ':', 2) AS INT) RANGE BETWEEN 10 PRECEDING AND 1 PRECEDING) as traffic_density
                      
        FROM raw_pbp p
        LEFT JOIN team_nets n ON p.game_id = n.game_id AND p.period = n.period AND p.event_owner_team_id = n.event_owner_team_id
        WINDOW w AS (PARTITION BY p.game_id ORDER BY p.event_id)
    ),
    shots_only AS (
        SELECT * FROM pbp_with_prev
        WHERE event_type IN ('shot-on-goal', 'goal') AND x_coord IS NOT NULL
    ),
    shift_seconds AS (
        SELECT 
            game_id, team_id, period,
            CAST(SPLIT_PART(start_time, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(start_time, ':', 2) AS INT) as start_sec,
            CAST(SPLIT_PART(end_time, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(end_time, ':', 2) AS INT) as end_sec
        FROM raw_shifts
    ),
    shot_ids AS (
        SELECT game_id, period, event_id, time_seconds, home_team_id, away_team_id FROM shots_only
    ),
    strength_agg AS (
        SELECT 
            s.event_id,
            COALESCE(SUM(CASE WHEN sh.team_id = s.home_team_id THEN 1 ELSE 0 END), 0) as home_skaters,
            COALESCE(SUM(CASE WHEN sh.team_id = s.away_team_id THEN 1 ELSE 0 END), 0) as away_skaters
        FROM shot_ids s
        LEFT JOIN shift_seconds sh
          ON s.game_id = sh.game_id 
         AND s.period = sh.period
         AND s.time_seconds >= sh.start_sec 
         AND s.time_seconds < sh.end_sec
        GROUP BY s.event_id
    )
    SELECT s.*, st.home_skaters, st.away_skaters 
    FROM shots_only s
    LEFT JOIN strength_agg st ON s.event_id = st.event_id
    """
    
    df = conn.execute(query).df()
    
    if len(df) == 0:
        logger.warning("No valid shots found in database. Exiting feature build.")
        return
        
    logger.info(f"Extracted {len(df)} shots. Computing mathematical features...")
    
    df['net_x'] = df['net_x'].fillna(89) 
    df['shot_distance'] = np.sqrt((df['net_x'] - df['x_coord'])**2 + df['y_coord']**2)
    df['shot_angle'] = np.where(df['shot_distance'] > 0, np.abs(np.arcsin(df['y_coord'] / df['shot_distance'])) * 180 / np.pi, 0)
                                
    df['time_since_last_event'] = df['time_seconds'] - df['prev_time_seconds']
    df['time_since_last_stoppage'] = df['time_seconds'] - df['last_stoppage_time']
    
    # Fill NA sequence events first so np.where doesn't crash on pd.NA
    df['time_since_last_event'] = df['time_since_last_event'].fillna(999).astype(float)
    df['time_since_last_stoppage'] = df['time_since_last_stoppage'].fillna(999).astype(float)
    df['prev_x'] = df['prev_x'].fillna(89).astype(float) # Default prev location to net to avoid NA math
    df['prev_y'] = df['prev_y'].fillna(0).astype(float)
    
    # New final features
    df['prev_distance'] = np.sqrt((df['net_x'] - df['prev_x'])**2 + df['prev_y']**2)
    df['prev_angle'] = np.where(df['prev_distance'] > 0, np.abs(np.arcsin(df['prev_y'] / df['prev_distance'])) * 180 / np.pi, 0)
    df['delta_angle'] = np.abs(df['shot_angle'] - df['prev_angle']).fillna(0)
    
    df['distance_from_prev'] = np.sqrt((df['x_coord'] - df['prev_x'])**2 + (df['y_coord'] - df['prev_y'])**2)
    df['puck_speed'] = np.where(df['time_since_last_event'] > 0, df['distance_from_prev'] / df['time_since_last_event'], 0)
    df['puck_speed'] = df['puck_speed'].fillna(0)
    
    df['score_differential'] = np.where(df['event_owner_team_id'] == df['home_team_id'], 
                                        df['home_goals'] - df['away_goals'], 
                                        df['away_goals'] - df['home_goals'])
    
    # Compute strength state (e.g., 5v5, 5v4) from goalie's perspective
    def get_strength_state(row):
        # We only care about shots reaching the goalie.
        # If the shooting team is the away team, they are attacking the home goalie.
        if pd.isna(row['home_skaters']) or pd.isna(row['away_skaters']):
            return 'unknown'
        if row['event_owner_team_id'] == row['away_team_id']:
            return f"{int(row['away_skaters'])}v{int(row['home_skaters'])}" # Attacking v Defending
        else:
            return f"{int(row['home_skaters'])}v{int(row['away_skaters'])}"
            
    df['strength_state'] = df.apply(get_strength_state, axis=1)
    
    df['royal_road_cross'] = ((df['time_since_last_event'] <= 3) & (np.sign(df['y_coord']) != np.sign(df['prev_y'])) & (np.sign(df['prev_x']) == np.sign(df['net_x']))).astype(int)
    
    def extract_is_empty_net(details):
        if pd.isna(details): return 0
        try:
            d = json.loads(details)
            if 'goalieInNetId' not in d or d['goalieInNetId'] is None: return 1
            return 0
        except: return 0
            
    df['is_empty_net'] = df['details_json'].apply(extract_is_empty_net)
    
    def extract_shot_type(details, existing_shot_type):
        if pd.notna(existing_shot_type): return existing_shot_type
        if pd.isna(details): return 'unknown'
        try:
            d = json.loads(details)
            return d.get('shotType', 'unknown')
        except: return 'unknown'
            
    df['shot_type'] = df.apply(lambda row: extract_shot_type(row['details_json'], row['shot_type']), axis=1)
    df['shot_type'] = df['shot_type'].fillna('unknown')
    
    df['sequence_2_events'] = df['sequence_2_events'].fillna('None')
    df['prev_event_type'] = df['prev_event_type'].fillna('None')
    df['time_since_last_event'] = df['time_since_last_event'].fillna(999) 
    df['time_since_last_stoppage'] = df['time_since_last_stoppage'].fillna(999)
    
    df['is_goal'] = (df['event_type'] == 'goal').astype(int)
    
    df = df[(df['is_empty_net'] == 1) | (df['goalie_in_net_id'].notna())]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow', index=False)
    logger.info(f"Saved {len(df)} engineered features to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    build_xg_features()
