import pandas as pd
import numpy as np
import duckdb
from pathlib import Path

def prepare_card_data():
    # Load all metrics
    scored = pd.read_parquet("data/processed/scored_shots.parquet")
    dsis = pd.read_parquet("data/processed/dsis_posteriors.parquet")
    proj = pd.read_parquet("data/processed/kalman_projections.parquet")
    map_df = pd.read_parquet("data/processed/goalie_map.parquet")
    base_metrics = pd.read_parquet("data/processed/goalie_base_metrics.parquet")
    
    df_2026 = scored[scored['season'] == 20252026]
    
    # Calculate seasonal stats
    conn = duckdb.connect()
    query = """
    SELECT 
        goalie_in_net_id as goalie_id,
        COUNT(*) as shots,
        SUM(is_goal) as goals,
        SUM(xg_prob) - SUM(is_goal) as gsax_2_0,
        AVG(traffic_density) as avg_traffic,
        AVG(puck_speed) as avg_speed,
        AVG(delta_angle) as avg_movement
    FROM df_2026
    GROUP BY 1
    HAVING COUNT(*) >= 200
    """
    stats = conn.execute(query).df()
    
    # Merge with True Talent and Projections
    stats = stats.merge(dsis[['goalie_id', 'dsis_true_talent_gsax_per_shot']], on='goalie_id', how='left')
    stats = stats.merge(proj[['goalie_id', 'proj_1yr_talent_per_shot']], on='goalie_id', how='left')
    
    # Merge with RCI from Base Metrics (take most recent season available, e.g., 20252026)
    base_2026 = base_metrics[base_metrics['season'] == 20252026]
    stats = stats.merge(base_2026[['goalie_in_net_id', 'rci']].rename(columns={'goalie_in_net_id': 'goalie_id'}), on='goalie_id', how='left')
    
    # Add Names
    name_map = map_df.set_index('goalie_id')['goalie_name'].to_dict()
    stats['Goalie'] = stats['goalie_id'].map(name_map).fillna(stats['goalie_id'].astype(str))
    
    # CALCULATE PERCENTILES
    # Rank 100 = Best in league
    cols_to_percentile = ['gsax_2_0', 'dsis_true_talent_gsax_per_shot', 'avg_traffic', 'avg_speed', 'avg_movement', 'rci']
    for col in cols_to_percentile:
        stats[f'{col}_percentile'] = stats[col].rank(pct=True) * 100
        
    # Zone-based stats for heatmap (9-zone grid on the net)
    # We'll use adjusted_y and adjusted_z (if we had z)
    # Since we only have x,y (rink), we'll use shot zones (High Slot, Low Slot, Point, etc.)
    # Actually, JFresh cards often use a 5-zone net (Top Left, Top Right, Low Left, Low Right, Five-Hole)
    # We don't have puck-hit-net coordinates in the PBP usually.
    # We have shot origin. We'll use shot origin heatmaps (High Danger, Medium Danger, Low Danger).
    
    # Save the prepared stats
    output_path = "data/processed/card_stats.parquet"
    stats.to_parquet(output_path, index=False)
    print(f"Saved card stats for {len(stats)} goalies to {output_path}")

if __name__ == "__main__":
    prepare_card_data()
