import pandas as pd
import requests
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def fetch_name(player_id):
    """Fetch Name from NHL API with exponential backoff."""
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                first = data.get('firstName', {}).get('default', '')
                last = data.get('lastName', {}).get('default', '')
                return player_id, f"{first} {last}".strip()
            elif response.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
        except Exception as e:
            logger.warning(f"Error fetching {player_id}: {e}")
        time.sleep(0.5)
    return player_id, str(player_id)

def build_goalie_map():
    # Load all unique goalie IDs from scored shots
    scored_path = "data/processed/scored_shots.parquet"
    if not Path(scored_path).exists():
        logger.error(f"Scored shots not found at {scored_path}")
        return

    df = pd.read_parquet(scored_path)
    unique_ids = df['goalie_in_net_id'].dropna().unique().astype(int)
    
    logger.info(f"Fetching names for {len(unique_ids)} goalies...")
    
    name_map = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_name, unique_ids))
        for res in results:
            name_map.append(res)
            
    map_df = pd.DataFrame(name_map, columns=['goalie_id', 'goalie_name'])
    
    output_path = "data/processed/goalie_map.parquet"
    map_df.to_parquet(output_path, index=False)
    logger.info(f"Saved goalie map to {output_path}")

if __name__ == "__main__":
    build_goalie_map()
