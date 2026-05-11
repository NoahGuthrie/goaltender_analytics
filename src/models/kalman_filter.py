import pandas as pd
import numpy as np
import logging
from pathlib import Path
from pykalman import KalmanFilter
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# Reference: A starter faces ~1700-1900 shots per season
REFERENCE_SHOTS = 1800  # What we consider a "full season" for measurement noise scaling

def run_kalman_projections(
    ages_path="data/processed/goalie_ages.parquet",
    drift_path="data/processed/age_curve_drift.parquet",
    output_path="data/processed/kalman_projections.parquet"
):
    logger.info("Loading goalie seasonal true talent and age data...")
    df = pd.read_parquet(ages_path)
    drift_df = pd.read_parquet(drift_path)
    
    # Only run the filter on goalies with at least 2 seasons of data
    df = df.sort_values(['goalie_id', 'season'])
    valid_goalies = df.groupby('goalie_id').size()
    valid_goalies = valid_goalies[valid_goalies >= 2].index
    df = df[df['goalie_id'].isin(valid_goalies)].copy()
    
    logger.info("Initializing Kalman Filter parameters...")
    
    # Create the drift lookup function
    drift_func = interp1d(drift_df['age'], drift_df['yearly_drift'], kind='linear', fill_value='extrapolate')
    
    # --- GLOBAL PARAMETERS (Scaled for GSAx/Shot) ---
    # Process noise: how much can true talent genuinely change year-over-year?
    # Scale: (0.01 / 30)^2 ≈ 0.00001
    PROCESS_NOISE = 0.00001  
    
    # Base measurement noise for a full-season starter (1800+ shots)
    # Scale: (0.04 / 900) ≈ 0.00004
    BASE_MEASUREMENT_NOISE = 0.00004
    
    # Initial state covariance: high uncertainty for rookies
    INITIAL_STATE_COVARIANCE = [[0.001]]
    
    results = []
    career_trajectories = []  # For visualization
    
    logger.info(f"Filtering {len(valid_goalies)} goalies with games-played shrinkage...")
    
    for goalie_id in valid_goalies:
        goalie_data = df[df['goalie_id'] == goalie_id].copy()
        
        # Observations: Seasonal True Talent (DSIS-Isolated GSAx 2.0 per Shot)
        measurements = goalie_data['seasonal_true_talent'].values
        ages = goalie_data['exact_age'].values
        shots = goalie_data['shots_faced'].values  # Precision shrinkage using actual shots faced
        n_timesteps = len(ages)
        
        # --- Age Curve Drift (time-varying transition offset) ---
        transition_offsets = np.zeros((n_timesteps, 1))
        for i in range(1, n_timesteps):
            age_diff = ages[i] - ages[i-1]
            yearly_rate = drift_func(ages[i-1])
            transition_offsets[i, 0] = yearly_rate * age_diff
        
        # --- Shots Faced Shrinkage (time-varying observation covariance) ---
        # A starter who faces 1800 shots gets the base noise.
        # This forces the filter to DISTRUST short-sample hot streaks.
        observation_covariances = np.zeros((n_timesteps, 1, 1))
        for i in range(n_timesteps):
            shots_weight = max(REFERENCE_SHOTS / max(shots[i], 1), 1.0)
            observation_covariances[i, 0, 0] = BASE_MEASUREMENT_NOISE * shots_weight
        
        kf = KalmanFilter(
            transition_matrices=[[1.0]],
            observation_matrices=[[1.0]],
            initial_state_mean=[0.0],  # Start rookies at league average
            initial_state_covariance=INITIAL_STATE_COVARIANCE,
            transition_offsets=transition_offsets,
            transition_covariance=[[PROCESS_NOISE]],
            observation_covariance=observation_covariances
        )
        
        # Run the Kalman Filter (forward pass only — no future leakage)
        filtered_state_means, filtered_state_covariances = kf.filter(measurements)
        
        # Store career trajectory for visualization
        for i in range(n_timesteps):
            career_trajectories.append({
                'goalie_id': goalie_id,
                'season': goalie_data.iloc[i]['season'],
                'age': ages[i],
                'shots_faced': shots[i],
                'observed_talent': measurements[i],
                'kalman_talent': filtered_state_means[i, 0],
                'kalman_uncertainty': np.sqrt(filtered_state_covariances[i, 0, 0])
            })
        
        # --- PROJECTIONS ---
        last_mean = filtered_state_means[-1, 0]
        last_cov = filtered_state_covariances[-1, 0, 0]
        last_age = ages[-1]
        last_shots = shots[-1]
        
        # 1-Year Projection
        drift_1yr = drift_func(last_age)
        proj_1yr_mean = last_mean + drift_1yr
        proj_1yr_cov = last_cov + PROCESS_NOISE
        
        # 3-Year Projection (drift compounds, noise compounds)
        drift_3yr = sum(drift_func(last_age + y) for y in range(3))
        proj_3yr_mean = last_mean + drift_3yr
        proj_3yr_cov = last_cov + (PROCESS_NOISE * 3)
        
        results.append({
            'goalie_id': goalie_id,
            'latest_season': goalie_data.iloc[-1]['season'],
            'current_age': round(last_age, 1),
            'latest_shots_faced': int(shots[-1]),
            'current_filtered_talent_per_shot': round(last_mean, 6),
            'current_uncertainty': round(np.sqrt(last_cov), 6),
            'proj_1yr_talent_per_shot': round(proj_1yr_mean, 6),
            'proj_1yr_ci_lower': round(proj_1yr_mean - 1.28 * np.sqrt(proj_1yr_cov), 6),  # 80% CI
            'proj_1yr_ci_upper': round(proj_1yr_mean + 1.28 * np.sqrt(proj_1yr_cov), 6),
            'proj_3yr_talent_per_shot': round(proj_3yr_mean, 6),
        })
    
    proj_df = pd.DataFrame(results)
    
    # Load Name Map
    map_path = "data/processed/goalie_map.parquet"
    if Path(map_path).exists():
        name_map = pd.read_parquet(map_path).set_index('goalie_id')['goalie_name'].to_dict()
        proj_df['Goalie'] = proj_df['goalie_id'].map(name_map).fillna(proj_df['goalie_id'].astype(str))
    else:
        proj_df['Goalie'] = proj_df['goalie_id'].astype(str)
        
    proj_df = proj_df.sort_values('proj_1yr_talent_per_shot', ascending=False)
    
    # Save trajectories
    traj_df = pd.DataFrame(career_trajectories)
    traj_df.to_parquet(str(output_path).replace('projections.parquet', 'kalman_trajectories.parquet'), index=False)
    
    # Save projections
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    proj_df.to_parquet(output_path, index=False)
    logger.info(f"Saved Kalman Projections to {output_path}")
    
    # --- Output ---
    print("\n" + "="*80)
    print("   KALMAN FILTER PROJECTIONS — NEXT SEASON (GSAx 2.0 / Game)")
    print("   Based on DSIS-Isolated True Talent with Games-Played Shrinkage")
    print("="*80)
    
    active = proj_df[proj_df['latest_season'] == 20252026].copy()
    active = active.sort_values('proj_1yr_talent_per_shot', ascending=False)
    
    print(f"\n{'Rank':<5} {'Goalie':<22} {'Age':<5} {'Shots':<6} {'Current':<9} {'Proj 1Y':<9} {'80% CI (Per Game)':<20}")
    print("-" * 85)
    
    for i, (_, row) in enumerate(active.head(15).iterrows(), 1):
        # Convert to standardized per-game (x30 shots) for human readability
        current_pg = row['current_filtered_talent_per_shot'] * 30
        proj_pg = row['proj_1yr_talent_per_shot'] * 30
        ci_pg = f"[{row['proj_1yr_ci_lower']*30:+.3f}, {row['proj_1yr_ci_upper']*30:+.3f}]"
        
        print(f"{i:<5} {row['Goalie']:<22} {row['current_age']:<5.0f} {row['latest_shots_faced']:<6} "
              f"{current_pg:+.4f}  {proj_pg:+.4f}  {ci_pg:<20}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_kalman_projections()
