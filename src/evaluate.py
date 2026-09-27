import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(
    y_true_log: pd.Series, 
    y_pred_log: np.ndarray
) -> Dict[str, float]:
    """
    Calculates evaluation metrics in both log scale and real dollar values.
    """
    # Inverse log transformation (expm1) to convert back to USD
    y_true_dollars = np.expm1(y_true_log)
    y_pred_dollars = np.expm1(y_pred_log)

    metrics = {
        "log_rmse": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "log_r2": float(r2_score(y_true_log, y_pred_log)),
        "real_mae": float(mean_absolute_error(y_true_dollars, y_pred_dollars)),
        "real_rmse": float(np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars)))
    }
    return metrics


def plot_evaluation_summary(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test_log: pd.Series, 
    save_path: str = None
) -> None:
    """Generates and displays feature importances and actual vs predicted plots."""
    y_pred_log = model.predict(X_test)
    y_test_dollars = np.expm1(y_test_log)
    y_pred_dollars = np.expm1(y_pred_log)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    # 1. Feature Importances
    importances = pd.Series(
        model.feature_importances_, index=X_test.columns
    ).sort_values(ascending=False).head(12)
    
    sns.barplot(x=importances.values, y=importances.index, palette='crest', ax=axes[0])
    axes[0].set_title("Top 12 Most Important Features", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Relative Importance Score", fontsize=10)

    # 2. Actual vs. Predicted Prices
    sns.scatterplot(
        x=y_test_dollars / 1e3, y=y_pred_dollars / 1e3, 
        alpha=0.5, color='teal', ax=axes[1]
    )
    axes[1].plot([0, 3000], [0, 3000], '--r', linewidth=2)
    axes[1].set_title("Actual vs. Predicted Price ($ Thousands)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Actual Price ($K)", fontsize=10)
    axes[1].set_ylabel("Predicted Price ($K)", fontsize=10)
    axes[1].set_xlim(0, 3000)
    axes[1].set_ylim(0, 3000)

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✓ Evaluation plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Evaluate module loaded successfully.")