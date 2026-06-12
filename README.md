# Customer Retention Prediction

Predicts customer churn risk using the Instacart Market Basket Analysis dataset. Combines RFM-based feature engineering with machine learning to identify at-risk customers and segment them into actionable risk tiers.

## Problem Statement

Retail businesses lose significant revenue to customer churn. This project builds a binary classifier to identify customers likely to churn based on their order history and purchasing behavior - without requiring an explicit cancellation event.

**Churn Definition:** A customer is labeled churned if their last recorded order gap is 28+ days, a standard RFM recency threshold used in retail analytics.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.797 | 0.661 | 0.578 | 0.617 | 0.848 |
| XGBoost | 0.801 | 0.701 | 0.518 | 0.596 | **0.866** |

XGBoost achieves **0.866 ROC-AUC** on 206,209 customers. Out of the full customer base, 27,147 customers (13.2%) are flagged as high churn risk.

## Dataset

[Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) - 3M+ orders from 206k customers.

Downloaded via:
```python
import kagglehub
path = kagglehub.dataset_download("psparks/instacart-market-basket-analysis")
```

## Features Engineered

**Order behavior** - total orders, avg/std days between orders, most common order day and hour

**Basket behavior** - avg basket size, basket size std, reorder rate, unique products, total items

**Diversity** - department diversity, top department

**Trend** - basket size trend (first half vs second half of order history)

## Project Structure

```
customer-retention/
├── src/
│   ├── eda.py                  - exploratory analysis, 5 charts
│   ├── feature_engineering.py  - builds customer-level feature matrix
│   ├── train.py                - trains LR + XGBoost, generates SHAP plots
│   └── predict.py              - scores customers, outputs risk tiers
├── outputs/
│   ├── charts/                 - EDA, confusion matrices, ROC, SHAP plots
│   └── models/                 - saved model files (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/ishan-singh98/customer-retention.git
cd customer-retention
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run in order:

```bash
python src/eda.py
python src/feature_engineering.py
python src/train.py
python src/predict.py
```

`predict.py` outputs `churn_predictions.csv` with churn probability and risk tier (Low/Medium/High) for every customer.

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Kagglehub