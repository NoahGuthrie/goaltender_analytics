import pandas as pd
import requests
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def fetch_name(player_id):
    """Fetch Name from NHL API with exponential backoff."""
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    
    # We will try up to 5 times, with increasingly longer delays to respect rate limits.
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                first = data.get('firstName', {}).get('default', '')
                last = data.get('lastName', {}).get('default', '')
                return f"{first} {last}".strip()
            elif response.status_code == 429:
                sleep_time = 2 ** (attempt + 2) # 4, 8, 16, 32, 64 seconds
                logger.warning(f"Rate limited on {player_id}. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            else:
                logger.warning(f"Failed to fetch {player_id}, status code: {response.status_code}")
                # For non-429 errors, wait a bit and retry
                time.sleep(2)
        except Exception as e:
            logger.warning(f"Error fetching {player_id}: {e}")
            time.sleep(2)
            
    return str(player_id) # Fallback to ID if all attempts fail

def fix_missing_names():
    map_path = "data/processed/goalie_map.parquet"
    if not Path(map_path).exists():
        logger.error(f"Goalie map not found at {map_path}")
        return

    df = pd.read_parquet(map_path)
    
    # Identify rows where the name is just the ID string
    missing_mask = df['goalie_id'].astype(str) == df['goalie_name']
    missing_ids = df[missing_mask]['goalie_id'].tolist()
    
    logger.info(f"Found {len(missing_ids)} missing names. Fetching sequentially to avoid rate limits...")
    
    updates = 0
    for idx, row in df[missing_mask].iterrows():
        gid = row['goalie_id']
        logger.info(f"Fetching {gid} ({updates + 1}/{len(missing_ids)})...")
        
        name = fetch_name(gid)
        if name != str(gid):
            df.at[idx, 'goalie_name'] = name
            updates += 1
            
        # Hard sleep between requests to be very polite to the API
        time.sleep(1.5)
        
    if updates > 0:
        logger.info(f"Successfully fetched {updates} names. Saving updated map...")
        df.to_parquet(map_path, index=False)
        logger.info("Done.")
    else:
        logger.info("No names were updated.")

if __name__ == "__main__":
    fix_missing_names()
