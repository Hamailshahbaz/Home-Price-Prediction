# src/train.py
import os
import sys
from sklearn.model_selection import train_test_split

# Ensure src directory is in Python path when executed directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import load_raw_data, clean_raw_data
from feature_engineering import build_feature_pipeline
from models import train_model, save_model_artifacts, evaluate_cross_validation
from evaluate import calculate_metrics, plot_evaluation_summary


def run_training_pipeline(
    raw_data_path: str = "../data/raw/data.csv",
    processed_data_path: str = "../data/processed/cleaned_housing_data.csv",
    models_dir: str = "../models",
    reports_dir: str = "../reports"
):
    """Executes end-to-end data ingestion, feature engineering, model training, and evaluation."""
    print("Starting Model Training Pipeline...\n")

    # Step 1: Ingestion & Cleaning
    print("[1/5] Ingesting and cleaning raw dataset...")
    raw_df = load_raw_data(raw_data_path)
    clean_df = clean_raw_data(raw_df)

    # Step 2: Feature Engineering
    print("[2/5] Engineering features and building target variable...")
    processed_df = build_feature_pipeline(clean_df)
    
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    processed_df.to_csv(processed_data_path, index=False)
    print(f"      Processed dataset saved to: {processed_data_path}")

    # Step 3: Train / Test Split
    print("[3/5] Splitting features and target...")
    drop_cols = ['date', 'street', 'city', 'statezip', 'country', 'price', 'log_price']
    X = processed_df.drop(columns=drop_cols)
    y = processed_df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Step 4: Cross-Validation & Training
    print("[4/5] Evaluating via 5-Fold CV and training final XGBoost model...")
    cv_scores = evaluate_cross_validation(X=X_train, y=y_train)
    print(f"      Cross-Validation Mean Log RMSE: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    model = train_model(X_train, y_train)

    # Step 5: Test Evaluation & Artifact Export
    print("[5/5] Evaluating model on test set and exporting artifacts...")
    y_pred_log = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred_log)

    print("\n" + "="*45)
    print("       FINAL TEST SET EVALUATION METRICS       ")
    print("="*45)
    print(f"  Log-Price RMSE : {metrics['log_rmse']:.4f}")
    print(f"  Log-Price R²   : {metrics['log_r2']:.4f}")
    print(f"  Real Price MAE : ${metrics['real_mae']:,.2f} USD")
    print("="*45 + "\n")

    # Export Model & Plots
    model_path, features_path = save_model_artifacts(model, X.columns.tolist(), output_dir=models_dir)
    print(f"✓ Model artifact saved to: {model_path}")
    print(f"✓ Features list saved to:  {features_path}")

    plot_path = os.path.join(reports_dir, "model_evaluation_summary.png")
    plot_evaluation_summary(model, X_test, y_test, save_path=plot_path)

    print("\nTraining Pipeline Completed Successfully!")


if __name__ == "__main__":
    run_training_pipeline()