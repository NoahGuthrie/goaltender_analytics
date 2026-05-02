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
    
    # 1. Base table with window functions and true attacking net resolution
    query = """
    WITH team_nets AS (
        -- Determine which net each team is attacking per period
        SELECT 
            game_id, 
            period, 
            event_owner_team_id,
            CASE WHEN avg(x_coord) > 0 THEN 89 ELSE -89 END as net_x
        FROM raw_pbp
        WHERE event_type IN ('shot-on-goal', 'goal', 'missed-shot')
          AND x_coord IS NOT NULL
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
            -- Time logic (convert MM:SS to seconds)
            CAST(SPLIT_PART(p.time_in_period, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(p.time_in_period, ':', 2) AS INT) as time_seconds,
            
            -- Window functions for previous events
            LAG(p.event_type, 1) OVER w as prev_event_type,
            LAG(p.x_coord, 1) OVER w as prev_x,
            LAG(p.y_coord, 1) OVER w as prev_y,
            LAG(CAST(SPLIT_PART(p.time_in_period, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(p.time_in_period, ':', 2) AS INT), 1) OVER w as prev_time_seconds,
            LAG(p.event_type, 2) OVER w || '->' || LAG(p.event_type, 1) OVER w as sequence_2_events,
            
            -- Shot sequences (count shots in the 4 seconds prior)
            COUNT(CASE WHEN p.event_type IN ('shot-on-goal', 'goal') THEN 1 END) 
                OVER (PARTITION BY p.game_id 
                      ORDER BY CAST(SPLIT_PART(p.time_in_period, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(p.time_in_period, ':', 2) AS INT) 
                      RANGE BETWEEN 4 PRECEDING AND 1 PRECEDING) as shot_sequence_num,
                      
            -- Traffic density (events in offensive zone in last 10s)
            COUNT(CASE WHEN SIGN(p.x_coord) = SIGN(n.net_x) AND p.event_type IN ('hit', 'blocked-shot', 'faceoff', 'giveaway', 'takeaway', 'shot-on-goal', 'missed-shot') THEN 1 END)
                OVER (PARTITION BY p.game_id 
                      ORDER BY CAST(SPLIT_PART(p.time_in_period, ':', 1) AS INT) * 60 + CAST(SPLIT_PART(p.time_in_period, ':', 2) AS INT) 
                      RANGE BETWEEN 10 PRECEDING AND 1 PRECEDING) as traffic_density
                      
        FROM raw_pbp p
        LEFT JOIN team_nets n ON p.game_id = n.game_id AND p.period = n.period AND p.event_owner_team_id = n.event_owner_team_id
        WINDOW w AS (PARTITION BY p.game_id ORDER BY p.event_id)
    )
    SELECT * FROM pbp_with_prev
    WHERE event_type IN ('shot-on-goal', 'goal') 
      AND x_coord IS NOT NULL
    """
    
    df = conn.execute(query).df()
    
    if len(df) == 0:
        logger.warning("No valid shots found in database. Exiting feature build.")
        return
        
    logger.info(f"Extracted {len(df)} shots. Computing mathematical features...")
    
    # 2. Mathematical features
    # Handle rare cases where net_x is NaN (e.g. no shots recorded in a period by a team)
    df['net_x'] = df['net_x'].fillna(89) 
    
    df['shot_distance'] = np.sqrt((df['net_x'] - df['x_coord'])**2 + df['y_coord']**2)
    
    # Avoid division by zero for angle. Use arcsin to get angle in degrees.
    # A shot from straight on (y=0) is 0 degrees. A shot from the goal line (x=89, y=42) is 90 degrees.
    df['shot_angle'] = np.where(df['shot_distance'] > 0, 
                                np.abs(np.arcsin(df['y_coord'] / df['shot_distance'])) * 180 / np.pi, 
                                0)
                                
    df['time_since_last_event'] = df['time_seconds'] - df['prev_time_seconds']
    
    # 3. Proxies for missing tracking data
    # Cross-ice proxy (royal road cross): Previous event < 3s ago, y_coord flipped sign, previous event was in offensive zone
    df['royal_road_cross'] = (
        (df['time_since_last_event'] <= 3) & 
        (np.sign(df['y_coord']) != np.sign(df['prev_y'])) &
        (np.sign(df['prev_x']) == np.sign(df['net_x'])) # Previous event was in offensive zone
    ).astype(int)
    
    # Empty net extraction from details_json
    def extract_is_empty_net(details):
        if pd.isna(details): return 0
        try:
            d = json.loads(details)
            if 'goalieInNetId' not in d or d['goalieInNetId'] is None:
                return 1
            return 0
        except:
            return 0
            
    df['is_empty_net'] = df['details_json'].apply(extract_is_empty_net)
    
    # Extract shot type safely
    def extract_shot_type(details, existing_shot_type):
        if pd.notna(existing_shot_type): return existing_shot_type
        if pd.isna(details): return 'unknown'
        try:
            d = json.loads(details)
            return d.get('shotType', 'unknown')
        except:
            return 'unknown'
            
    df['shot_type'] = df.apply(lambda row: extract_shot_type(row['details_json'], row['shot_type']), axis=1)
    df['shot_type'] = df['shot_type'].fillna('unknown')
    
    # 4. Fill NA sequence events
    df['sequence_2_events'] = df['sequence_2_events'].fillna('None')
    df['prev_event_type'] = df['prev_event_type'].fillna('None')
    df['time_since_last_event'] = df['time_since_last_event'].fillna(999) # Treat missing as long time ago
    
    # Target variable: 1 for goal, 0 for save
    df['is_goal'] = (df['event_type'] == 'goal').astype(int)
    
    # Drop rows without a valid goalie ID if they are not empty netters
    df = df[(df['is_empty_net'] == 1) | (df['goalie_in_net_id'].notna())]
    
    # Write to Parquet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow', index=False)
    logger.info(f"Saved {len(df)} engineered features to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    build_xg_features()
