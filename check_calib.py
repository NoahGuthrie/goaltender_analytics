import duckdb
import pandas as pd

conn = duckdb.connect()

query = """
WITH season_calib AS (
    SELECT 
        season, 
        SUM(is_goal) / SUM(xg_prob) as calib_factor
    FROM 'data/processed/scored_shots.parquet'
    GROUP BY season
),
calibrated_shots AS (
    SELECT 
        s.*,
        s.xg_prob * c.calib_factor as xg_prob_calibrated
    FROM 'data/processed/scored_shots.parquet' s
    JOIN season_calib c ON s.season = c.season
)
SELECT 
    goalie_in_net_id as goalie_id,
    CASE WHEN event_owner_team_id = home_team_id THEN away_team_id ELSE home_team_id END as team_id,
    COUNT(*) as shots_faced,
    SUM(xg_prob_calibrated) as xG_against,
    SUM(is_goal) as goals_against,
    SUM(xg_prob_calibrated) - SUM(is_goal) as gsax_2_0
FROM calibrated_shots
WHERE goalie_in_net_id IN (8478048, 8482193, 8477484, 8471734) -- Rangers goalies (Shesty, Garand, Martin, Quick)
  AND season = 20252026
  AND is_empty_net = 0
GROUP BY 1, 2
ORDER BY gsax_2_0 DESC
"""

df = conn.execute(query).df()
print(df.to_string(index=False))
