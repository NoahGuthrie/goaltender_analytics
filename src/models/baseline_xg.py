import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from src.models.evaluation import evaluate_model

logger = logging.getLogger(__name__)

def train_baseline(data_path="data/processed/xg_features.parquet", model_dir="data/models"):
    logger.info("Loading features...")
    df = pd.read_parquet(data_path)
    
    if len(df) < 50:
        logger.warning("Not enough data to train effectively. Need more than 50 rows.")
    
    # §1.1 Features: Distance, Angle, Shot Type, Empty Net
    features = ['shot_distance', 'shot_angle', 'shot_type', 'is_empty_net']
    target = 'is_goal'
    
    X = df[features]
    y = df[target]
    
    # Stratification fails if we don't have at least 2 instances of each class
    stratify = y if y.sum() >= 2 and (len(y) - y.sum()) >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    
    # Preprocessing
    numeric_features = ['shot_distance', 'shot_angle']
    categorical_features = ['shot_type']
    
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SklearnPipeline
    
    numeric_transformer = SklearnPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # For is_empty_net
    )
    
    # Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    
    logger.info("Training baseline Logistic Regression model...")
    clf.fit(X_train, y_train)
    
    logger.info("Evaluating baseline model...")
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_prob, model_name="Baseline_LR")
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, Path(model_dir) / 'baseline_lr.joblib')
    logger.info(f"Model saved to {Path(model_dir) / 'baseline_lr.joblib'}")
    
    return clf, metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_baseline()
