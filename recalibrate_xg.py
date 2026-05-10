import duckdb
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def recalibrate():
    conn = duckdb.connect()
    data_path = 'data/processed/scored_shots.parquet'
    temp_path = 'data/processed/scored_shots_temp.parquet'
    
    logging.info("Calculating seasonal calibration factors...")
    
    # Check original
    orig = conn.execute(f"SELECT SUM(xg_prob) as xg, SUM(is_goal) as goals FROM '{data_path}'").fetchone()
    logging.info(f"Original: xG={orig[0]:.1f}, Goals={orig[1]:.1f}")
    
    conn.execute(f"""
    COPY (
        WITH season_calib AS (
            SELECT 
                season, 
                SUM(is_goal) / SUM(xg_prob) as calib_factor
            FROM '{data_path}'
            GROUP BY season
        )
        SELECT 
            s.* EXCLUDE (xg_prob),
            s.xg_prob * c.calib_factor as xg_prob
        FROM '{data_path}' s
        JOIN season_calib c ON s.season = c.season
    ) TO '{temp_path}' (FORMAT PARQUET)
    """)
    
    # Verify new
    new = conn.execute(f"SELECT SUM(xg_prob) as xg, SUM(is_goal) as goals FROM '{temp_path}'").fetchone()
    logging.info(f"Calibrated: xG={new[0]:.1f}, Goals={new[1]:.1f}")
    
    # Replace file
    os.remove(data_path)
    os.rename(temp_path, data_path)
    logging.info("Recalibration complete and saved.")

if __name__ == "__main__":
    recalibrate()
