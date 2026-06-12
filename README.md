# Customer Retention Prediction

Predicts customer churn using the Instacart Market Basket Analysis dataset.

## Problem Statement
Identify customers at risk of churning based on their order history and purchasing behavior.

## Approach
- Feature engineering from order history (recency, frequency, basket trends)
- Baseline logistic regression + XGBoost classifier
- SHAP values for model explainability

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn

## Project Structure
- `src/feature_engineering.py` - builds customer-level features
- `src/train.py` - model training and evaluation
- `src/predict.py` - scoring new customers
- `outputs/charts/` - EDA and SHAP plots

## Dataset
[Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)