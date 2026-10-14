"""
SHAP feature importance and evaluation plots for the XGBoost model.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from train_baseline import (
    load_split_data, train_xgboost, FEATURE_COLS, TARGET_COL, BASE_DIR,
)

FIG_DIR = BASE_DIR / "results/figures"
SEQ_LEN = 7


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_split_data()
    model, preds = train_xgboost(train, test)

    # LSTM predictions saved by train_lstm.py (covers test days after the first 7)
    lstm_preds = np.load(BASE_DIR / "results/lstm_preds.npy")
    lstm_dates = test["date"].iloc[SEQ_LEN:]

    # SHAP feature importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test[FEATURE_COLS])

    plt.figure()
    shap.summary_plot(shap_values, test[FEATURE_COLS], show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=150)
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    print("\nFeature importance (mean |SHAP|):")
    for name, val in sorted(zip(FEATURE_COLS, mean_abs), key=lambda x: -x[1]):
        print(f"  {name:22s} {val:.2f}")

    # Predicted vs actual
    plt.figure(figsize=(10, 4))
    plt.plot(test["date"], test[TARGET_COL], label="Actual")
    plt.plot(test["date"], preds, label="XGBoost")
    plt.plot(lstm_dates, lstm_preds, label="LSTM", alpha=0.7)
    plt.plot(test["date"], test["AQI"], label="Persistence", linestyle="--", alpha=0.5)
    plt.ylabel("AQI")
    plt.legend()
    plt.title("Next-day AQI: actual vs predicted (test set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close()

    # Absolute error over time
    plt.figure(figsize=(10, 3))
    plt.plot(test["date"], np.abs(test[TARGET_COL].values - preds))
    plt.ylabel("Absolute error")
    plt.title("XGBoost absolute error over time")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "error_over_time.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()