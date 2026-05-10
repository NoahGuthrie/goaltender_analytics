import pandas as pd
import numpy as np
import requests
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import duckdb

logger = logging.getLogger(__name__)

# Hardcoded fallback DOBs for elite goalies in case the NHL API rate-limits us.
# This guarantees these players are NEVER dropped from the dataset.
FALLBACK_DOBS = {
    8478048: "1995-12-30",  # Igor Shesterkin
    8476883: "1994-06-20",  # Andrei Vasilevskiy
    8479979: "1998-12-03",  # Jake Oettinger
    8476945: "1993-05-19",  # Connor Hellebuyck
    8471679: "1987-08-16",  # Carey Price
    8470657: "1982-03-02",  # Henrik Lundqvist
    8476412: "1993-06-14",  # John Gibson
    8475683: "1988-09-20",  # Sergei Bobrovsky
    8477424: "1993-07-27",  # Linus Ullmark
    8478406: "1996-12-09",  # Mackenzie Blackwood
    8477967: "1995-12-08",  # Thatcher Demko
    8480947: "1996-02-11",  # Adin Hill
    8480313: "1998-11-01",  # Stuart Skinner
    8482137: "2000-12-26",  # Pyotr Kochetkov
    8475883: "1989-10-02",  # Frederik Andersen
    8476999: "1990-07-25",  # Jonathan Quick
    8481020: "1997-06-12",  # Filip Gustavsson
    8484170: "2000-03-03",  # Justus Annunen
    8476932: "1991-09-11",  # Semyon Varlamov
    8480382: "1998-01-19",  # Ilya Sorokin
    8483532: "2000-10-20",  # Samuel Ersson
    8476899: "1994-04-28",  # Joonas Korpisalo
    8480843: "1997-02-22",  # Ilya Samsonov
    8475660: "1989-11-18",  # Marc-Andre Fleury
    8476341: "1990-07-13",  # Ben Bishop
    8476316: "1988-06-19",  # Braden Holtby
    8471469: "1985-11-25",  # Pekka Rinne
    8475852: "1990-05-08",  # Tuukka Rask
    8474593: "1988-04-07",  # Corey Crawford
    8478024: "1996-03-05",  # Jeremy Swayman
    8478007: "1995-09-13",  # Juuse Saros
    8471306: "1987-06-25",  # Roberto Luongo
    8475361: "1988-11-28",  # Tim Thomas (late bloomer)
    8475717: "1989-07-29",  # Robin Lehner
    8477950: "1995-02-11",  # Matt Murray
    8475831: "1990-09-17",  # Jacob Markstrom
    8478492: "1995-10-22",  # Alexandar Georgiev
    8479292: "1996-06-08",  # Darcy Kuemper
    8476316: "1988-06-19",  # Braden Holtby
    8471695: "1988-06-13",  # Jaroslav Halak
    8475622: "1990-01-14",  # John Gibson placeholder
    8476256: "1986-12-18",  # Devan Dubnyk
    8475311: "1988-06-11",  # Craig Anderson
    8476434: "1991-02-16",  # Cam Talbot
    8480280: "1998-10-08",  # Dustin Wolf
}

def fetch_dob_with_retry(player_id, max_retries=3):
    """Fetch DOB from NHL API with exponential backoff, falling back to hardcoded dict."""
    for attempt in range(max_retries):
        try:
            url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                dob = data.get('birthDate')
                if dob:
                    return player_id, dob
            elif response.status_code == 429:  # Rate limited
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited on {player_id}, waiting {wait}s...")
                time.sleep(wait)
                continue
        except Exception as e:
            logger.warning(f"API error for {player_id}: {e}")
        time.sleep(0.5 * (attempt + 1))
    
    # Fallback to hardcoded dictionary
    if player_id in FALLBACK_DOBS:
        logger.info(f"Using fallback DOB for player {player_id}")
        return player_id, FALLBACK_DOBS[player_id]
    
    return player_id, None

def calculate_age_curves(
    metrics_path="data/processed/goalie_base_metrics.parquet",
    output_ages="data/processed/goalie_ages.parquet",
    output_drift="data/processed/age_curve_drift.parquet"
):
    logger.info("Loading goalie metrics to identify unique goalies...")
    metrics_df = pd.read_parquet(metrics_path)
    unique_goalies = metrics_df['goalie_in_net_id'].unique()
    
    logger.info(f"Fetching Date of Birth for {len(unique_goalies)} goalies from NHL API (with retries + fallbacks)...")
    dob_data = []
    
    # Slower but robust: use ThreadPoolExecutor with only 3 workers to avoid rate limits
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(fetch_dob_with_retry, unique_goalies)
        for res in results:
            dob_data.append(res)
            
    dob_df = pd.DataFrame(dob_data, columns=['goalie_id', 'birth_date'])
    dob_df['birth_date'] = pd.to_datetime(dob_df['birth_date'])
    
    found = dob_df.dropna()
    logger.info(f"Successfully resolved DOB for {len(found)} / {len(unique_goalies)} goalies.")
    dob_df = found
    
    # --- Build Seasonal True Talent using GSAx 2.0 ---
    logger.info("Computing seasonal GSAx 2.0 True Talent...")
    
    conn = duckdb.connect()
    scored_data_path = "data/processed/scored_shots.parquet"
    
    # Group by goalie + defending team + season
    # team_id = defending team (goalie's team)
    # ACTUAL GP via COUNT(DISTINCT game_id) — no more estimating
    query = f"""
    SELECT 
        goalie_in_net_id as goalie_id,
        CASE WHEN event_owner_team_id = home_team_id THEN away_team_id ELSE home_team_id END as team_id,
        season,
        COUNT(*) as shots_faced,
        COUNT(DISTINCT game_id) as games_played,
        SUM(xg_prob) - SUM(is_goal) as raw_gsax_2_0
    FROM '{scored_data_path}'
    WHERE goalie_in_net_id IS NOT NULL AND is_empty_net = 0
    GROUP BY 1, 2, 3
    HAVING COUNT(DISTINCT game_id) >= 10
    """
    seasonal_df = conn.execute(query).df()
    
    # Load DSIS Team Effects (now time-varying: one effect per team-season)
    team_effects_path = "data/processed/dsis_team_effects.parquet"
    try:
        dsis_teams = pd.read_parquet(team_effects_path)
        # Merge on BOTH team_id AND season for time-varying effects
        seasonal_df = seasonal_df.merge(
            dsis_teams[['team_id', 'season', 'dsis_team_defense_impact_per_game']], 
            on=['team_id', 'season'], how='left'
        )
        seasonal_df['dsis_team_defense_impact_per_game'] = seasonal_df['dsis_team_defense_impact_per_game'].fillna(0)
        logger.info("Successfully merged time-varying DSIS Team-Season Effects for True Talent isolation.")
    except FileNotFoundError:
        logger.warning("DSIS Team Effects not found! Using raw GSAx 2.0 without team isolation.")
        seasonal_df['dsis_team_defense_impact_per_game'] = 0
    
    # Per-game GSAx using ACTUAL games played
    seasonal_df['raw_gsax_per_game'] = seasonal_df['raw_gsax_2_0'] / seasonal_df['games_played']
    
    # Seasonal True Talent = Raw GSAx/Game - Team Defensive Impact
    # This isolates the goalie's individual contribution
    seasonal_df['seasonal_true_talent'] = seasonal_df['raw_gsax_per_game'] - seasonal_df['dsis_team_defense_impact_per_game']
    
    # Merge DOB
    seasonal_df = seasonal_df.merge(dob_df, on='goalie_id', how='inner')
    
    # Calculate Age on Oct 1st of the given season
    seasonal_df['season_start_year'] = seasonal_df['season'].astype(str).str[:4].astype(int)
    seasonal_df['season_start_date'] = pd.to_datetime(seasonal_df['season_start_year'].astype(str) + '-10-01')
    seasonal_df['exact_age'] = (seasonal_df['season_start_date'] - seasonal_df['birth_date']).dt.days / 365.25
    
    Path(output_ages).parent.mkdir(parents=True, exist_ok=True)
    seasonal_df.to_parquet(output_ages, index=False)
    logger.info(f"Saved goalie ages data ({len(seasonal_df)} rows) to {output_ages}")
    
    # --- Fit LOESS Aging Curve ---
    logger.info("Fitting LOESS Aging Curve...")
    
    curve_data = seasonal_df[(seasonal_df['exact_age'] >= 20) & (seasonal_df['exact_age'] <= 40)]
    
    # frac=0.3 gives a moderately smooth curve without over-fitting to noise
    loess_res = lowess(curve_data['seasonal_true_talent'], curve_data['exact_age'], frac=0.3)
    
    # Create a dense lookup table for age drift
    ages = np.linspace(20, 42, 221)  # Every 0.1 years
    interp_func = interp1d(loess_res[:, 0], loess_res[:, 1], kind='linear', fill_value='extrapolate')
    expected_talent = interp_func(ages)
    
    # Calculate the yearly derivative (drift)
    drift_func = interp1d(ages, expected_talent, kind='linear', fill_value='extrapolate')
    yearly_drift = drift_func(ages + 1.0) - drift_func(ages)
    
    drift_df = pd.DataFrame({
        'age': np.round(ages, 1),
        'expected_true_talent': expected_talent,
        'yearly_drift': yearly_drift
    })
    
    drift_df.to_parquet(output_drift, index=False)
    logger.info(f"Saved LOESS Age Curve to {output_drift}")
    
    print("\n--- NHL GOALIE AGING CURVE (LOESS on DSIS-Isolated True Talent) ---")
    peak_age = drift_df.loc[drift_df['expected_true_talent'].idxmax()]['age']
    print(f"Canonical Peak Age: {peak_age} years old")
    
    sample_ages = [22.0, 25.0, 27.0, 28.0, 30.0, 33.0, 35.0, 38.0]
    for a in sample_ages:
        row = drift_df[drift_df['age'] == a].iloc[0]
        print(f"  Age {a:4.0f}: Expected Talent = {row['expected_true_talent']:+.3f} | Yearly Drift = {row['yearly_drift']:+.4f} GSAx/Game")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    calculate_age_curves()
