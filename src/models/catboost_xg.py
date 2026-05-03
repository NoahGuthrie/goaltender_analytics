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
        'shot_distance', 'shot_angle', 'shot_type', 'is_empty_net',
        'time_since_last_event', 'prev_event_type', 'sequence_2_events',
        'shot_sequence_num', 'traffic_density', 'royal_road_cross',
        'time_since_last_stoppage', 'strength_state'
    ]
    target = 'is_goal'
    
    # Identify categorical features for CatBoost
    cat_features = ['shot_type', 'prev_event_type', 'sequence_2_events', 'strength_state']
    
    # Cast categoricals to string (CatBoost requires string or int for categoricals)
    for col in cat_features:
        df[col] = df[col].astype(str)
        
    X = df[features]
    y = df[target]
    
    # Due to small sample size for the smoke test, we'll do a simple train/test split.
    # In full production, this would be GroupKFold by 'season' for Leave-One-Season-Out (LOSO).
    from sklearn.model_selection import train_test_split
    stratify = y if y.sum() >= 2 and (len(y) - y.sum()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    
    logger.info("Training CatBoost Context-Enriched model...")
    
    # Load tuned parameters if they exist
    import json
    params_path = Path("data/models/tune/best_catboost_params.json")
    if params_path.exists():
        logger.info(f"Loading tuned parameters from {params_path}")
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        clf = CatBoostClassifier(
            iterations=500,
            auto_class_weights='Balanced',
            cat_features=cat_features,
            verbose=100,
            random_state=42,
            **best_params
        )
    else:
        logger.info("Using default parameters...")
        clf = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            cat_features=cat_features,
            auto_class_weights='Balanced',
            verbose=100,
            random_state=42
        )
    
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
    
    logger.info("Evaluating CatBoost model...")
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_prob, model_name="CatBoost_ContextEnriched")
    
    # SHAP values (CatBoost supports SHAP natively but we need the raw model)
    try:
        logger.info("Generating SHAP summary plot...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
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
