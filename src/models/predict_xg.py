import pandas as pd
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

def score_dataset(
    features_path="data/processed/xg_features.parquet",
    model_path="data/models/catboost_xg.joblib",
    output_path="data/processed/scored_shots.parquet"
):
    logger.info("Loading xG features dataset...")
    df = pd.read_parquet(features_path)
    
    logger.info("Loading trained CatBoost model...")
    clf = joblib.load(model_path)
    
    features = [
        'shot_distance', 'shot_angle', 'shot_type', 'is_empty_net',
        'time_since_last_event', 'prev_event_type', 'sequence_2_events',
        'shot_sequence_num', 'traffic_density', 'royal_road_cross',
        'time_since_last_stoppage', 'strength_state',
        'score_differential', 'puck_speed', 'delta_angle'
    ]
    
    cat_features = ['shot_type', 'prev_event_type', 'sequence_2_events', 'strength_state']
    
    logger.info("Preparing data for inference...")
    for col in cat_features:
        df[col] = df[col].astype(str)
        
    X = df[features]
    
    logger.info(f"Predicting exact goal probabilities for {len(df)} shots...")
    # Get probabilities for class 1 (is_goal)
    df['xg_prob'] = clf.predict_proba(X)[:, 1]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow', index=False)
    
    logger.info(f"Saved scored dataset to {output_path}")
    logger.info(f"Average xG: {df['xg_prob'].mean():.4f} | Actual Goal Rate: {df['is_goal'].mean():.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    score_dataset()
