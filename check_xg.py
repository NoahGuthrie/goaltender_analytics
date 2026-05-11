import duckdb
import pandas as pd

conn = duckdb.connect()

df = conn.execute("""
SELECT 
    season,
    COUNT(*) as shots, 
    SUM(is_goal) as total_goals, 
    SUM(xg_prob) as total_xg, 
    SUM(is_goal)*1.0/COUNT(*) as avg_goal_rate, 
    SUM(xg_prob)*1.0/COUNT(*) as avg_xg_prob 
FROM 'data/processed/scored_shots.parquet' 
GROUP BY season
ORDER BY season DESC
LIMIT 5
""").df()

print(df.to_string(index=False))
