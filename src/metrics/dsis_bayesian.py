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
    
    logger.info("Aggregating shots to GAME level for MCMC sampling...")
    
    # GAME-LEVEL aggregation: each row = one goalie's performance in one game
    # This gives us ~40-80 observations per team-season instead of ~3
    # team_id = the DEFENDING team (the goalie's team)
    query = f"""
    SELECT 
        goalie_in_net_id as goalie_id,
        CASE WHEN event_owner_team_id = home_team_id THEN away_team_id ELSE home_team_id END as team_id,
        season,
        game_id,
        COUNT(*) as shots_faced,
        SUM(xg_prob) - SUM(is_goal) as game_gsax,
        AVG(traffic_density) as avg_traffic,
        AVG(puck_speed) as avg_speed
    FROM '{scored_data_path}'
    WHERE goalie_in_net_id IS NOT NULL 
      AND is_empty_net = 0
    GROUP BY 1, 2, 3, 4
    HAVING COUNT(*) > 5
    """
    
    df = conn.execute(query).df()
    logger.info(f"Aggregated to {len(df)} goalie-game observations.")
    
    # Filter to goalies with enough games to stabilize
    valid_goalies = df.groupby('goalie_id').size()
    valid_goalies = valid_goalies[valid_goalies >= 20].index
    df = df[df['goalie_id'].isin(valid_goalies)].copy()
    
    # Standardize covariates
    df['traffic_std'] = (df['avg_traffic'] - df['avg_traffic'].mean()) / df['avg_traffic'].std()
    df['speed_std'] = (df['avg_speed'] - df['avg_speed'].mean()) / df['avg_speed'].std()
    
    # Create indices
    goalie_idx, goalies = pd.factorize(df['goalie_id'])
    
    # TIME-VARYING: compound (team, season) index
    df['team_season'] = df['team_id'].astype(str) + '_' + df['season'].astype(str)
    team_season_idx, team_seasons = pd.factorize(df['team_season'])
    
    season_idx, seasons = pd.factorize(df['season'])
    
    n_goalies = len(goalies)
    n_team_seasons = len(team_seasons)
    n_seasons = len(seasons)
    n_obs = len(df)
    
    # Count observations per team-season to verify coverage
    ts_obs = df.groupby('team_season').size()
    logger.info(f"Model dimensions: {n_goalies} goalies, {n_team_seasons} team-seasons, {n_seasons} seasons, {n_obs} observations")
    logger.info(f"Observations per team-season: mean={ts_obs.mean():.1f}, min={ts_obs.min()}, max={ts_obs.max()}")
    
    logger.info("Defining PyMC Hierarchical Model (DSIS v2 — Game Level)...")
    
    # Monkey-patch convergence check (numpy 2.4 / xarray crash)
    import pymc.stats.convergence as _conv
    _original_check = _conv.run_convergence_checks
    _conv.run_convergence_checks = lambda *a, **kw: []
    
    try:
        with pm.Model() as dsis_model:
            idx_goalie = pm.Data("idx_goalie", goalie_idx)
            idx_ts = pm.Data("idx_ts", team_season_idx)
            idx_season = pm.Data("idx_season", season_idx)
            traffic = pm.Data("traffic", df['traffic_std'].values)
            speed = pm.Data("speed", df['speed_std'].values)
            
            # Fixed Effects
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            beta_traffic = pm.Normal("beta_traffic", mu=0, sigma=1)
            beta_speed = pm.Normal("beta_speed", mu=0, sigma=1)
            
            # Random Effects
            sigma_goalie = pm.HalfNormal("sigma_goalie", sigma=1)
            sigma_ts = pm.HalfNormal("sigma_ts", sigma=1)
            sigma_season = pm.HalfNormal("sigma_season", sigma=1)
            
            z_goalie = pm.Normal("z_goalie", mu=0, sigma=1, shape=n_goalies)
            z_ts = pm.Normal("z_ts", mu=0, sigma=1, shape=n_team_seasons)
            z_season = pm.Normal("z_season", mu=0, sigma=1, shape=n_seasons)
            
            goalie_effect = pm.Deterministic("goalie_effect", z_goalie * sigma_goalie)
            team_season_effect = pm.Deterministic("team_season_effect", z_ts * sigma_ts)
            season_effect = pm.Deterministic("season_effect", z_season * sigma_season)
            
            mu = (
                alpha 
                + beta_traffic * traffic 
                + beta_speed * speed 
                + goalie_effect[idx_goalie] 
                + team_season_effect[idx_ts] 
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
    
    # --- Extract Goalie Effects ---
    goalie_samples = trace.posterior["goalie_effect"].values
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
    
    # --- Extract Time-Varying Team-Season Effects ---
    ts_samples = trace.posterior["team_season_effect"].values
    ts_flat = ts_samples.reshape(-1, n_team_seasons)
    ts_means = ts_flat.mean(axis=0)
    ts_stds = ts_flat.std(axis=0)
    
    ts_parts = pd.DataFrame({'team_season': team_seasons})
    ts_parts[['team_id', 'season']] = ts_parts['team_season'].str.split('_', expand=True)
    ts_parts['team_id'] = ts_parts['team_id'].astype(int)
    ts_parts['season'] = ts_parts['season'].astype(int)
    ts_parts['dsis_team_defense_impact_per_game'] = ts_means
    ts_parts['dsis_team_std'] = ts_stds
    
    team_output = str(output_path).replace('posteriors.parquet', 'team_effects.parquet')
    ts_parts[['team_id', 'season', 'dsis_team_defense_impact_per_game', 'dsis_team_std']].to_parquet(team_output, index=False)
    logger.info(f"Saved DSIS team-season effects to {team_output}")
    
    # --- Print Results ---
    names = {
        8478048: "Shesterkin", 8476883: "Vasilevskiy", 8479979: "Oettinger",
        8476945: "Hellebuyck", 8471679: "Price", 8470657: "Lundqvist",
        8476412: "Gibson", 8475683: "Bobrovsky", 8477424: "Ullmark",
        8477967: "Demko", 8480313: "Skinner", 8475883: "Andersen",
        8480382: "Sorokin", 8478024: "Swayman", 8478007: "Saros",
    }
    summary['Name'] = summary['goalie_id'].map(names).fillna(summary['goalie_id'].astype(str))
    
    print("\n--- TOP 15 TRUE TALENT GOALIES (DSIS v2 — Game-Level, Time-Varying) ---")
    print(summary.head(15)[['Name', 'dsis_true_talent_gsax_per_game', 'dsis_std_dev', 'dsis_hdi_lower', 'dsis_hdi_upper']].to_string(index=False))
    
    # Show Rangers by season
    rangers = ts_parts[ts_parts['team_id'] == 3].sort_values('season')
    if len(rangers) > 0:
        print("\n--- NY RANGERS DEFENSIVE IMPACT BY SEASON ---")
        print(rangers[['season', 'dsis_team_defense_impact_per_game']].to_string(index=False))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_dsis_model()
