import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import duckdb
import logging
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore", module="pytensor")

logger = logging.getLogger(__name__)

def run_dsis_model(scored_data_path="data/processed/scored_shots.parquet", output_path="data/processed/dsis_posteriors.parquet"):
    conn = duckdb.connect()
    
    logger.info("Aggregating shots to Game level for MCMC sampling...")
    
    query = f"""
    SELECT 
        goalie_in_net_id as goalie_id,
        CASE WHEN event_owner_team_id = home_team_id THEN away_team_id ELSE home_team_id END as team_id,
        season,
        COUNT(*) as shots_faced,
        SUM(xg_prob) as expected_goals,
        SUM(is_goal) as actual_goals,
        SUM(xg_prob) - SUM(is_goal) as game_gsax,
        AVG(traffic_density) as avg_traffic,
        AVG(puck_speed) as avg_speed
    FROM '{scored_data_path}'
    WHERE goalie_in_net_id IS NOT NULL 
      AND is_empty_net = 0
    GROUP BY 1, 2, 3
    HAVING COUNT(*) > 10 -- Minimum 10 shots to count as a real game appearance
    """
    
    df = conn.execute(query).df()
    logger.info(f"Aggregated down to {len(df)} goalie-game performances.")
    
    # Filter to goalies with at least 20 career games in the dataset to stabilize the model
    valid_goalies = df.groupby('goalie_id').size()
    valid_goalies = valid_goalies[valid_goalies >= 20].index
    df = df[df['goalie_id'].isin(valid_goalies)].copy()
    
    # Standardize covariates
    df['traffic_std'] = (df['avg_traffic'] - df['avg_traffic'].mean()) / df['avg_traffic'].std()
    df['speed_std'] = (df['avg_speed'] - df['avg_speed'].mean()) / df['avg_speed'].std()
    
    # Create categorical indices for PyMC
    goalie_idx, goalies = pd.factorize(df['goalie_id'])
    team_idx, teams = pd.factorize(df['team_id'])
    season_idx, seasons = pd.factorize(df['season'])
    
    coords = {
        "goalie": goalies,
        "team": teams,
        "season": seasons,
        "obs_id": np.arange(len(df))
    }
    
    logger.info("Defining PyMC Hierarchical Model (DSIS)...")
    
    with pm.Model(coords=coords) as dsis_model:
        # Data
        idx_goalie = pm.Data("idx_goalie", goalie_idx, dims="obs_id")
        idx_team = pm.Data("idx_team", team_idx, dims="obs_id")
        idx_season = pm.Data("idx_season", season_idx, dims="obs_id")
        
        traffic = pm.Data("traffic", df['traffic_std'].values, dims="obs_id")
        speed = pm.Data("speed", df['speed_std'].values, dims="obs_id")
        
        y = pm.Data("y", df['game_gsax'].values, dims="obs_id")
        
        # Fixed Effects (Priors)
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta_traffic = pm.Normal("beta_traffic", mu=0, sigma=1)
        beta_speed = pm.Normal("beta_speed", mu=0, sigma=1)
        
        # Random Effects (Hierarchical Priors) - Non-centered parameterization for better sampling
        sigma_goalie = pm.HalfNormal("sigma_goalie", sigma=1)
        sigma_team = pm.HalfNormal("sigma_team", sigma=1)
        sigma_season = pm.HalfNormal("sigma_season", sigma=1)
        
        z_goalie = pm.Normal("z_goalie", mu=0, sigma=1, dims="goalie")
        z_team = pm.Normal("z_team", mu=0, sigma=1, dims="team")
        z_season = pm.Normal("z_season", mu=0, sigma=1, dims="season")
        
        goalie_effect = pm.Deterministic("goalie_effect", z_goalie * sigma_goalie, dims="goalie")
        team_effect = pm.Deterministic("team_effect", z_team * sigma_team, dims="team")
        season_effect = pm.Deterministic("season_effect", z_season * sigma_season, dims="season")
        
        # Expected Value
        mu = (
            alpha 
            + beta_traffic * traffic 
            + beta_speed * speed 
            + goalie_effect[idx_goalie] 
            + team_effect[idx_team] 
            + season_effect[idx_season]
        )
        
        # Error term
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=2)
        
        # Likelihood
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma_eps, observed=y, dims="obs_id")
        
        logger.info("Starting MCMC Sampling with Numpyro (JAX backend)...")
        trace = pm.sample(draws=1000, tune=1000, chains=2, cores=1, progressbar=False, return_inferencedata=True, nuts_sampler="numpyro")
        
    logger.info("Sampling complete! Extracting posterior true-talent estimates...")
    
    summary = az.summary(trace, var_names=["goalie_effect"], hdi_prob=0.95)
    summary = summary.reset_index()
    summary['goalie_id'] = summary['index'].apply(lambda x: int(float(re.search(r'\[(.*?)\]', x).group(1))))
    
    summary = summary.rename(columns={
        'mean': 'dsis_true_talent_gsax_per_game',
        'sd': 'dsis_std_dev',
        'hdi_2.5%': 'dsis_hdi_lower',
        'hdi_97.5%': 'dsis_hdi_upper'
    })
    
    summary = summary[['goalie_id', 'dsis_true_talent_gsax_per_game', 'dsis_std_dev', 'dsis_hdi_lower', 'dsis_hdi_upper']]
    summary = summary.sort_values('dsis_true_talent_gsax_per_game', ascending=False)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output_path, index=False)
    logger.info(f"Saved DSIS goalie posteriors to {output_path}")
    
    # Extract Team Effects
    team_summary = az.summary(trace, var_names=["team_effect"], hdi_prob=0.95)
    team_summary = team_summary.reset_index()
    team_summary['team_id'] = team_summary['index'].apply(lambda x: int(float(re.search(r'\[(.*?)\]', x).group(1))))
    team_summary = team_summary.rename(columns={'mean': 'dsis_team_defense_impact_per_game'})
    team_summary = team_summary[['team_id', 'dsis_team_defense_impact_per_game']]
    team_summary.to_parquet(str(output_path).replace('posteriors.parquet', 'team_effects.parquet'), index=False)
    logger.info("Saved DSIS team effects.")
    
    print("\n--- TOP 10 TRUE TALENT GOALIES (DSIS) ---")
    print(summary.head(10).to_string(index=False))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_dsis_model()
