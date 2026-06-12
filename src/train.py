import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
import shap

FEATURES_PATH = "outputs/customer_features.csv"
MODELS_PATH = "outputs/models"
CHARTS_PATH = "outputs/charts"
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(CHARTS_PATH, exist_ok=True)

FEATURE_COLS = [
    "total_orders",  "avg_days_between_orders",
    "std_days_between_orders", "most_common_dow", "most_common_hour",
    "avg_basket_size", "std_basket_size", "reorder_rate",
    "unique_products", "total_items", "dept_diversity",
    "top_department_id", "basket_size_trend"
]
TARGET = "churned"


def load_features():
    print("Loading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df[TARGET].mean():.2%}")
    return df


def evaluate_model(name, y_test, y_pred, y_prob):
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }


def plot_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Churned", "Churned"],
                yticklabels=["Not Churned", "Churned"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(CHARTS_PATH, f"confusion_matrix_{fname}.png"))
    plt.close()
    print(f"Saved: confusion_matrix_{fname}.png")


def plot_roc_curves(models_data):
    plt.figure(figsize=(8, 6))
    for name, y_test, y_prob in models_data:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, "roc_curves.png"))
    plt.close()
    print("Saved: roc_curves.png")


def plot_feature_importance(model, feature_cols):
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="steelblue")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, "feature_importance.png"))
    plt.close()
    print("Saved: feature_importance.png")


def plot_shap(model, X_test):
    print("\nGenerating SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:2000])

    plt.figure()
    shap.summary_plot(shap_values, X_test[:2000], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_PATH, "shap_summary.png"), bbox_inches="tight")
    plt.close()
    print("Saved: shap_summary.png")


def train():
    df = load_features()

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

    lr_metrics = evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob)
    plot_confusion_matrix("Logistic Regression", y_test, lr_pred)

    joblib.dump(lr, os.path.join(MODELS_PATH, "logistic_regression.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))
    print("Saved: logistic_regression.pkl, scaler.pkl")

    # XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    xgb_metrics = evaluate_model("XGBoost", y_test, xgb_pred, xgb_prob)
    plot_confusion_matrix("XGBoost", y_test, xgb_pred)
    plot_feature_importance(xgb, FEATURE_COLS)

    joblib.dump(xgb, os.path.join(MODELS_PATH, "xgboost.pkl"))
    print("Saved: xgboost.pkl")

    # ROC curves
    plot_roc_curves([
        ("Logistic Regression", y_test, lr_prob),
        ("XGBoost", y_test, xgb_prob)
    ])

    # SHAP
    plot_shap(xgb, X_test)

    # Summary
    print("\n--- Model Comparison ---")
    results = pd.DataFrame([lr_metrics, xgb_metrics])
    print(results.to_string(index=False))

    results.to_csv(os.path.join("outputs", "model_results.csv"), index=False)
    print("\nResults saved: outputs/model_results.csv")


if __name__ == "__main__":
    train()