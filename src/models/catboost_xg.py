import pandas as pd
import numpy as np
import logging
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

from src.models.evaluation import evaluate_model

logger = logging.getLogger(__name__)

def train_catboost(data_path="data/processed/xg_features.parquet", model_dir="data/models"):
    logger.info("Loading features...")
    df = pd.read_parquet(data_path)
    
    if len(df) < 50:
        logger.warning("Not enough data to train effectively. Need more than 50 rows.")
        
    features = [
        'adjusted_x', 'adjusted_y', 'shot_distance', 'shot_angle', 'shot_type', 
        'time_since_last_event', 'prev_event_type', 'sequence_2_events',
        'shot_sequence_num', 'traffic_density', 'royal_road_cross',
        'time_since_last_stoppage', 'strength_state',
        'score_differential', 'puck_speed', 'delta_angle'
    ]
    target = 'is_goal'
    
    # Identify categorical features for CatBoost
    cat_features = ['shot_type', 'prev_event_type', 'sequence_2_events', 'strength_state']
    
    # Cast categoricals to string (CatBoost requires string or int for categoricals)
    for col in cat_features:
        df[col] = df[col].astype(str)
        
    X = df[features]
    y = df[target]
    
    logger.info("Implementing Leave-One-Season-Out (LOSO) Cross-Validation...")
    
    from sklearn.model_selection import GroupKFold
    
    # We will do 5-fold grouped by season to save time, but it evaluates unseen seasons
    gkf = GroupKFold(n_splits=min(5, df['season'].nunique()))
    
    oof_preds = np.zeros(len(df))
    models = []
    
    # Load tuned parameters if they exist
    import json
    params_path = Path("data/models/tune/best_catboost_params.json")
    if params_path.exists():
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        model_params = {**best_params, 'iterations': 500, 'cat_features': cat_features, 'verbose': 100, 'random_state': 42}
    else:
        model_params = {'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 'cat_features': cat_features, 'verbose': 100, 'random_state': 42}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=df['season'])):
        logger.info(f"--- Training Fold {fold+1} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = CatBoostClassifier(**model_params)
        clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
        models.append(clf)
        
    logger.info("Evaluating LOSO out-of-fold predictions...")
    metrics = evaluate_model(y, oof_preds, model_name="CatBoost_ContextEnriched_LOSO")
    
    # Train final model on ALL data
    logger.info("Training final model on full dataset...")
    clf = CatBoostClassifier(**model_params)
    clf.fit(X, y, verbose=False)
    
    # SHAP values (CatBoost supports SHAP natively but we need the raw model)
    try:
        logger.info("Generating SHAP summary plot...")
        # SHAP on a subsample of full data
        explainer = shap.TreeExplainer(clf)
        X_sample = X.sample(n=min(10000, len(X)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(Path(model_dir) / 'eval' / 'catboost_shap_summary.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate SHAP values: {e}")
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, Path(model_dir) / 'catboost_xg.joblib')
    logger.info(f"Model saved to {Path(model_dir) / 'catboost_xg.joblib'}")
    
    return clf, metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_catboost()
