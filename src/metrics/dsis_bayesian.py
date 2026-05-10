import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

import pymc as pm
import pandas as pd
import numpy as np
import duckdb
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", module="pytensor")
warnings.filterwarnings("ignore", module="xarray")

logger = logging.getLogger(__name__)

def run_dsis_model(scored_data_path="data/processed/scored_shots.parquet", output_path="data/processed/dsis_posteriors.parquet"):
    conn = duckdb.connect()
    
    logger.info("Aggregating shots to Goalie-Team-Season level for MCMC sampling...")
    
    # CRITICAL: team_id = the DEFENDING team (the goalie's team), NOT the shooting team
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
    HAVING COUNT(*) > 10
    """
    
    df = conn.execute(query).df()
    logger.info(f"Aggregated down to {len(df)} goalie-team-season rows.")
    
    # Filter to goalies with enough data to stabilize the model
    valid_goalies = df.groupby('goalie_id').size()
    valid_goalies = valid_goalies[valid_goalies >= 3].index  # At least 3 season-records
    df = df[df['goalie_id'].isin(valid_goalies)].copy()
    logger.info(f"After goalie filter: {len(df)} rows, {df['goalie_id'].nunique()} goalies, {df['team_id'].nunique()} teams")
    
    # Standardize covariates
    df['traffic_std'] = (df['avg_traffic'] - df['avg_traffic'].mean()) / df['avg_traffic'].std()
    df['speed_std'] = (df['avg_speed'] - df['avg_speed'].mean()) / df['avg_speed'].std()
    
    # Create categorical indices for PyMC
    goalie_idx, goalies = pd.factorize(df['goalie_id'])
    team_idx, teams = pd.factorize(df['team_id'])
    season_idx, seasons = pd.factorize(df['season'])
    
    n_goalies = len(goalies)
    n_teams = len(teams)
    n_seasons = len(seasons)
    n_obs = len(df)
    
    logger.info(f"Model dimensions: {n_goalies} goalies, {n_teams} teams, {n_seasons} seasons, {n_obs} observations")
    
    logger.info("Defining PyMC Hierarchical Model (DSIS)...")
    
    # Monkey-patch: disable post-sampling convergence check (numpy 2.4 / xarray crash)
    import pymc.stats.convergence as _conv
    _original_check = _conv.run_convergence_checks
    _conv.run_convergence_checks = lambda *a, **kw: []
    
    try:
        with pm.Model() as dsis_model:
            # Data containers
            idx_goalie = pm.Data("idx_goalie", goalie_idx)
            idx_team = pm.Data("idx_team", team_idx)
            idx_season = pm.Data("idx_season", season_idx)
            traffic = pm.Data("traffic", df['traffic_std'].values)
            speed = pm.Data("speed", df['speed_std'].values)
            
            # Fixed Effects
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            beta_traffic = pm.Normal("beta_traffic", mu=0, sigma=1)
            beta_speed = pm.Normal("beta_speed", mu=0, sigma=1)
            
            # Random Effects — using shape= instead of dims= for numpyro compatibility
            sigma_goalie = pm.HalfNormal("sigma_goalie", sigma=1)
            sigma_team = pm.HalfNormal("sigma_team", sigma=1)
            sigma_season = pm.HalfNormal("sigma_season", sigma=1)
            
            z_goalie = pm.Normal("z_goalie", mu=0, sigma=1, shape=n_goalies)
            z_team = pm.Normal("z_team", mu=0, sigma=1, shape=n_teams)
            z_season = pm.Normal("z_season", mu=0, sigma=1, shape=n_seasons)
            
            goalie_effect = pm.Deterministic("goalie_effect", z_goalie * sigma_goalie)
            team_effect = pm.Deterministic("team_effect", z_team * sigma_team)
            season_effect = pm.Deterministic("season_effect", z_season * sigma_season)
            
            # Expected Value
            mu = (
                alpha 
                + beta_traffic * traffic 
                + beta_speed * speed 
                + goalie_effect[idx_goalie] 
                + team_effect[idx_team] 
                + season_effect[idx_season]
            )
            
            sigma_eps = pm.HalfNormal("sigma_eps", sigma=2)
            likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma_eps, observed=df['game_gsax'].values)
            
            logger.info("Starting MCMC Sampling with Numpyro (JAX backend)...")
            trace = pm.sample(draws=1000, tune=1000, chains=2, cores=1, 
                             progressbar=False, nuts_sampler="numpyro")
    finally:
        _conv.run_convergence_checks = _original_check
    
    logger.info("Sampling complete! Extracting posterior estimates...")
    
    # Extract from InferenceData via xarray
    goalie_samples = trace.posterior["goalie_effect"].values  # (n_chains, n_draws, n_goalies)
    goalie_flat = goalie_samples.reshape(-1, n_goalies)
    goalie_means = goalie_flat.mean(axis=0)
    goalie_stds = goalie_flat.std(axis=0)
    
    summary = pd.DataFrame({
        'goalie_id': goalies,
        'dsis_true_talent_gsax_per_game': goalie_means,
        'dsis_std_dev': goalie_stds,
        'dsis_hdi_lower': goalie_means - (1.96 * goalie_stds),
        'dsis_hdi_upper': goalie_means + (1.96 * goalie_stds)
    })
    summary = summary.sort_values('dsis_true_talent_gsax_per_game', ascending=False)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output_path, index=False)
    logger.info(f"Saved DSIS goalie posteriors to {output_path}")
    
    # Extract Team Effects
    team_samples = trace.posterior["team_effect"].values
    team_flat = team_samples.reshape(-1, n_teams)
    team_means = team_flat.mean(axis=0)
    team_stds = team_flat.std(axis=0)
    
    team_summary = pd.DataFrame({
        'team_id': teams,
        'dsis_team_defense_impact_per_game': team_means,
        'dsis_team_std': team_stds
    })
    team_output = str(output_path).replace('posteriors.parquet', 'team_effects.parquet')
    team_summary.to_parquet(team_output, index=False)
    logger.info(f"Saved DSIS team effects to {team_output}")
    
    # Print results
    names = {
        8478048: "Shesterkin", 8476883: "Vasilevskiy", 8479979: "Oettinger",
        8476945: "Hellebuyck", 8471679: "Price", 8470657: "Lundqvist",
        8476412: "Gibson", 8475683: "Bobrovsky", 8477424: "Ullmark",
        8477967: "Demko", 8480313: "Skinner", 8475883: "Andersen",
        8480382: "Sorokin", 8478024: "Swayman", 8478007: "Saros",
    }
    summary['Name'] = summary['goalie_id'].map(names).fillna(summary['goalie_id'].astype(str))
    
    print("\n--- TOP 15 TRUE TALENT GOALIES (DSIS) ---")
    print(summary.head(15)[['Name', 'dsis_true_talent_gsax_per_game', 'dsis_std_dev', 'dsis_hdi_lower', 'dsis_hdi_upper']].to_string(index=False))
    
    print("\n--- TOP 5 DEFENSIVE SYSTEMS ---")
    team_summary_sorted = team_summary.sort_values('dsis_team_defense_impact_per_game', ascending=False)
    print(team_summary_sorted.head(5).to_string(index=False))
    
    print("\n--- BOTTOM 5 DEFENSIVE SYSTEMS ---")
    print(team_summary_sorted.tail(5).to_string(index=False))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_dsis_model()
