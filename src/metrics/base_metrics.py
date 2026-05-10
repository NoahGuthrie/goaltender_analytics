import duckdb
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def calculate_metrics(scored_data_path="data/processed/scored_shots.parquet", output_path="data/processed/goalie_base_metrics.parquet"):
    conn = duckdb.connect()
    
    logger.info("Aggregating metrics via DuckDB...")
    
    query = f"""
    WITH scored_shots AS (
        SELECT * FROM '{scored_data_path}'
        WHERE goalie_in_net_id IS NOT NULL 
          AND is_empty_net = 0
    ),
    -- Rebound events: A save is made, another shot occurs within 3s and 30ft
    rebound_logic AS (
        SELECT 
            *,
            -- Look forward: for each shot, is the NEXT event a shot on goal within 3 seconds?
            LEAD(time_seconds, 1) OVER w as next_time,
            LEAD(event_type, 1) OVER w as next_event_type,
            LEAD(shot_distance, 1) OVER w as next_distance,
            LEAD(xg_prob, 1) OVER w as next_xg,
            LEAD(event_owner_team_id, 1) OVER w as next_team_id
        FROM scored_shots
        WINDOW w AS (PARTITION BY game_id, goalie_in_net_id ORDER BY event_id)
    ),
    rebound_events AS (
        SELECT 
            *,
            CASE WHEN 
                is_goal = 0 AND -- Was a save
                next_event_type IN ('shot-on-goal', 'goal') AND
                next_team_id = event_owner_team_id AND -- Rebound by same attacking team
                (next_time - time_seconds) <= 3 AND 
                next_distance <= 30
            THEN 1 ELSE 0 END as generated_rebound,
            
            -- If it generated a rebound, how dangerous was the rebound?
            CASE WHEN 
                is_goal = 0 AND 
                next_event_type IN ('shot-on-goal', 'goal') AND
                next_team_id = event_owner_team_id AND
                (next_time - time_seconds) <= 3 AND 
                next_distance <= 30
            THEN next_xg ELSE 0 END as generated_rebound_xg
        FROM rebound_logic
    ),
    -- Movement Demand Adjustment
    -- MDA = avg(Movement Demand on saves) - avg(Movement Demand on goals allowed)
    mda_calc AS (
        SELECT 
            *,
            delta_angle / (time_since_last_event + 0.1) as movement_demand
        FROM rebound_events
    ),
    goalie_season_agg AS (
        SELECT 
            goalie_in_net_id,
            season,
            COUNT(*) as shots_faced,
            SUM(is_goal) as actual_goals_allowed,
            SUM(xg_prob) as expected_goals,
            SUM(xg_prob) - SUM(is_goal) as gsax_cumulative,
            
            -- RCI logic
            SUM(CASE WHEN is_goal = 0 THEN 1 ELSE 0 END) as total_saves,
            SUM(generated_rebound) as rebounds_generated,
            SUM(generated_rebound_xg) as total_rebound_xg_danger,
            
            -- MDA logic
            AVG(CASE WHEN is_goal = 0 THEN movement_demand END) as avg_mda_saves,
            AVG(CASE WHEN is_goal = 1 THEN movement_demand END) as avg_mda_goals
        FROM mda_calc
        GROUP BY 1, 2
        HAVING COUNT(*) > 100 -- Filter out tiny samples
    )
    SELECT 
        *,
        -- RCI Sub-metrics
        CAST(rebounds_generated AS FLOAT) / NULLIF(total_saves, 0) as rebound_rate,
        total_rebound_xg_danger / NULLIF(rebounds_generated, 0) as rebound_danger,
        
        -- Final MDA
        COALESCE(avg_mda_saves, 0) - COALESCE(avg_mda_goals, 0) as mda
    FROM goalie_season_agg
    """
    
    df = conn.execute(query).df()
    
    # Calculate League Averages for RCI
    df['rr_x_rd'] = df['rebound_rate'] * df['rebound_danger'].fillna(0)
    league_avg_rr_rd = df['rr_x_rd'].mean()
    
    df['rci'] = 1 - (df['rr_x_rd'] / league_avg_rr_rd)
    
    # Sort by GSAx to find the best
    df = df.sort_values('gsax_cumulative', ascending=False)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} goalie seasons to {output_path}")
    
    # Verification print
    print("\n--- TOP 10 GSAx SEASONS ---")
    print(df[['goalie_in_net_id', 'season', 'shots_faced', 'expected_goals', 'actual_goals_allowed', 'gsax_cumulative']].head(10).to_string(index=False))
    
    print("\n--- BOTTOM 5 GSAx SEASONS ---")
    print(df[['goalie_in_net_id', 'season', 'shots_faced', 'expected_goals', 'actual_goals_allowed', 'gsax_cumulative']].tail(5).to_string(index=False))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    calculate_metrics()
