import optuna
import pandas as pd
import numpy as np
import logging
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def objective(trial, X, y, cat_features):
    # Parameter search space
    params = {
        'iterations': 500,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'auto_class_weights': 'Balanced',
        'verbose': 0,
        'random_state': 42
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = CatBoostClassifier(**params)
        clf.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=30)
        
        y_prob = clf.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, y_prob)
        scores.append(score)
        
    return np.mean(scores)

def tune_model(data_path="data/processed/xg_features.parquet", output_dir="data/models/tune"):
    logger.info("Loading features for tuning...")
    df = pd.read_parquet(data_path)
    
    features = [
        'shot_distance', 'shot_angle', 'shot_type', 'is_empty_net',
        'time_since_last_event', 'prev_event_type', 'sequence_2_events',
        'shot_sequence_num', 'traffic_density', 'royal_road_cross',
        'time_since_last_stoppage', 'strength_state'
    ]
    target = 'is_goal'
    
    cat_features = ['shot_type', 'prev_event_type', 'sequence_2_events', 'strength_state']
    
    for col in cat_features:
        df[col] = df[col].astype(str)
        
    X = df[features]
    y = df[target]
    
    # Subsample to speed up tuning if dataset is huge (> 200k rows)
    if len(df) > 200000:
        logger.info(f"Subsampling 200k rows from {len(df)} total for faster Optuna tuning...")
        sub_idx = np.random.choice(len(df), 200000, replace=False)
        X = X.iloc[sub_idx].reset_index(drop=True)
        y = y.iloc[sub_idx].reset_index(drop=True)
    
    logger.info("Starting Optuna study...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y, cat_features), n_trials=15)
    
    logger.info("Tuning complete.")
    logger.info(f"Best Trial: {study.best_trial.value}")
    logger.info(f"Best Params: {study.best_params}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / 'best_catboost_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    return study.best_params

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    tune_model()
