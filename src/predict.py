import os
import pandas as pd
import numpy as np
import joblib

FEATURES_PATH = "outputs/customer_features.csv"
MODELS_PATH = "outputs/models"
OUTPUT_PATH = "outputs"

FEATURE_COLS = [
    "total_orders", "avg_days_between_orders",
    "std_days_between_orders", "most_common_dow", "most_common_hour",
    "avg_basket_size", "std_basket_size", "reorder_rate",
    "unique_products", "total_items", "dept_diversity",
    "top_department_id", "basket_size_trend"
]


def load_models():
    print("Loading models...")
    xgb = joblib.load(os.path.join(MODELS_PATH, "xgboost.pkl"))
    lr = joblib.load(os.path.join(MODELS_PATH, "logistic_regression.pkl"))
    scaler = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
    return xgb, lr, scaler


def score_customers(df, xgb, lr, scaler):
    print(f"Scoring {len(df):,} customers...")
    X = df[FEATURE_COLS]

    # XGBoost scores
    xgb_prob = xgb.predict_proba(X)[:, 1]
    xgb_pred = xgb.predict(X)

    # Logistic Regression scores
    X_scaled = scaler.transform(X)
    lr_prob = lr.predict_proba(X_scaled)[:, 1]
    lr_pred = lr.predict(X_scaled)

    results = df[["user_id"]].copy()
    results["xgb_churn_probability"] = xgb_prob
    results["xgb_churn_prediction"] = xgb_pred
    results["lr_churn_probability"] = lr_prob
    results["lr_churn_prediction"] = lr_pred

    # Risk tier based on XGBoost probability
    results["risk_tier"] = pd.cut(
        results["xgb_churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )

    return results


def print_summary(results):
    print("\n--- Churn Risk Summary ---")
    print(f"Total customers scored: {len(results):,}")
    print(f"\nRisk Tier Distribution:")
    print(results["risk_tier"].value_counts().sort_index())
    print(f"\nAvg churn probability: {results['xgb_churn_probability'].mean():.2%}")
    print(f"\nTop 10 highest risk customers:")
    top10 = results.nlargest(10, "xgb_churn_probability")[
        ["user_id", "xgb_churn_probability", "risk_tier"]
    ]
    print(top10.to_string(index=False))


def predict():
    df = pd.read_csv(FEATURES_PATH)
    xgb, lr, scaler = load_models()

    results = score_customers(df, xgb, lr, scaler)
    print_summary(results)

    output_file = os.path.join(OUTPUT_PATH, "churn_predictions.csv")
    results.to_csv(output_file, index=False)
    print(f"\nPredictions saved: {output_file}")

    return results


if __name__ == "__main__":
    predict()