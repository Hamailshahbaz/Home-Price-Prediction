import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor


def get_default_xgboost_model() -> XGBRegressor:
    """Returns the pre-configured, tuned XGBoost regressor instance."""
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


def evaluate_cross_validation(
    model: Any, 
    X: pd.DataFrame, 
    y: pd.Series, 
    cv_splits: int = 5
) -> np.ndarray:
    """Performs K-Fold cross-validation and returns RMSE scores across folds."""
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    neg_mse_scores = cross_val_score(
        model, X, y, scoring='neg_mean_squared_error', cv=cv
    )
    return np.sqrt(-neg_mse_scores)


def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model: Any = None
) -> Any:
    """Fits the specified regressor on training data."""
    if model is None:
        model = get_default_xgboost_model()
    
    model.fit(X_train, y_train)
    return model


def save_model_artifacts(
    model: Any, 
    feature_names: list, 
    output_dir: str = "../models"
) -> Tuple[str, str]:
    """Saves the trained model and feature metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "xgboost_housing_model.pkl")
    features_path = os.path.join(output_dir, "model_features.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)
    
    return model_path, features_path


if __name__ == "__main__":
    print("Models module loaded successfully.")