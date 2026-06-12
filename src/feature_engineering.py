import os
import pandas as pd
import numpy as np

DATA_PATH = r"C:\Users\Owner\.cache\kagglehub\datasets\psparks\instacart-market-basket-analysis\versions\1"
OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_data():
    print("Loading data...")
    orders = pd.read_csv(os.path.join(DATA_PATH, "orders.csv"))
    order_products_prior = pd.read_csv(os.path.join(DATA_PATH, "order_products__prior.csv"))
    products = pd.read_csv(os.path.join(DATA_PATH, "products.csv"))
    departments = pd.read_csv(os.path.join(DATA_PATH, "departments.csv"))
    return orders, order_products_prior, products, departments


def build_churn_label(orders, churn_threshold_days=28):
    """
    Churn definition: customer's last recorded order gap >= 28 days.
    Based on RFM recency - customers who have gone quiet are at churn risk.
    """
    print(f"Building churn labels (threshold: {churn_threshold_days} days)...")
    prior_orders = orders[orders["eval_set"] == "prior"]

    last_gap = prior_orders.groupby("user_id")["days_since_prior_order"].last().reset_index()
    last_gap.columns = ["user_id", "days_since_last_order"]

    last_gap["churned"] = (last_gap["days_since_last_order"] >= churn_threshold_days).astype(int)

    print(f"Churn rate: {last_gap['churned'].mean():.2%}")
    return last_gap[["user_id", "churned"]]


def build_order_features(orders):
    """Recency, frequency, and order pattern features per customer."""
    print("Building order features...")
    prior_orders = orders[orders["eval_set"] == "prior"]

    freq = prior_orders.groupby("user_id")["order_number"].max().rename("total_orders")
    recency = prior_orders.groupby("user_id")["days_since_prior_order"].last().rename("days_since_last_order")
    avg_gap = prior_orders.groupby("user_id")["days_since_prior_order"].mean().rename("avg_days_between_orders")
    std_gap = prior_orders.groupby("user_id")["days_since_prior_order"].std().rename("std_days_between_orders")

    common_dow = prior_orders.groupby("user_id")["order_dow"].agg(
        lambda x: x.value_counts().index[0]
    ).rename("most_common_dow")

    common_hour = prior_orders.groupby("user_id")["order_hour_of_day"].agg(
        lambda x: x.value_counts().index[0]
    ).rename("most_common_hour")

    order_features = pd.concat(
        [freq, recency, avg_gap, std_gap, common_dow, common_hour], axis=1
    ).reset_index()

    return order_features


def build_basket_features(orders, order_products_prior, products, departments):
    """Basket size, reorder rate, product diversity features per customer."""
    print("Building basket features...")

    prior_orders = orders[orders["eval_set"] == "prior"][["order_id", "user_id"]]
    op = order_products_prior.merge(prior_orders, on="order_id")

    basket_size = op.groupby(["user_id", "order_id"])["product_id"].count().reset_index()
    basket_size.columns = ["user_id", "order_id", "basket_size"]
    avg_basket = basket_size.groupby("user_id")["basket_size"].mean().rename("avg_basket_size")
    std_basket = basket_size.groupby("user_id")["basket_size"].std().rename("std_basket_size")

    reorder_rate = op.groupby("user_id")["reordered"].mean().rename("reorder_rate")
    unique_products = op.groupby("user_id")["product_id"].nunique().rename("unique_products")
    total_items = op.groupby("user_id")["product_id"].count().rename("total_items")

    op_dept = op.merge(products[["product_id", "department_id"]], on="product_id")
    dept_diversity = op_dept.groupby("user_id")["department_id"].nunique().rename("dept_diversity")

    top_dept = op_dept.groupby(["user_id", "department_id"])["product_id"].count().reset_index()
    top_dept = top_dept.sort_values("product_id", ascending=False).drop_duplicates("user_id")
    top_dept = top_dept[["user_id", "department_id"]].rename(columns={"department_id": "top_department_id"})

    basket_features = pd.concat(
        [avg_basket, std_basket, reorder_rate, unique_products, total_items, dept_diversity], axis=1
    ).reset_index()
    basket_features = basket_features.merge(top_dept, on="user_id", how="left")

    return basket_features


def build_trend_features(orders, order_products_prior):
    """Trend features - is basket size growing or shrinking over time?"""
    print("Building trend features...")

    prior_orders = orders[orders["eval_set"] == "prior"][["order_id", "user_id", "order_number"]]
    op = order_products_prior.merge(prior_orders, on="order_id")

    basket_over_time = op.groupby(["user_id", "order_number"])["product_id"].count().reset_index()
    basket_over_time.columns = ["user_id", "order_number", "basket_size"]

    def basket_trend(group):
        group = group.sort_values("order_number")
        mid = len(group) // 2
        if mid == 0:
            return 0
        first_half = group.iloc[:mid]["basket_size"].mean()
        second_half = group.iloc[mid:]["basket_size"].mean()
        return second_half - first_half

    trend = basket_over_time.groupby("user_id").apply(basket_trend).rename("basket_size_trend").reset_index()
    return trend


def build_features():
    orders, order_products_prior, products, departments = load_data()

    churn_labels = build_churn_label(orders)
    order_features = build_order_features(orders)
    basket_features = build_basket_features(orders, order_products_prior, products, departments)
    trend_features = build_trend_features(orders, order_products_prior)

    print("Merging features...")
    features = churn_labels \
        .merge(order_features, on="user_id", how="left") \
        .merge(basket_features, on="user_id", how="left") \
        .merge(trend_features, on="user_id", how="left")

    features["std_days_between_orders"] = features["std_days_between_orders"].fillna(0)
    features["std_basket_size"] = features["std_basket_size"].fillna(0)
    features["days_since_last_order"] = features["days_since_last_order"].fillna(
        features["days_since_last_order"].median()
    )

    output_file = os.path.join(OUTPUT_PATH, "customer_features.csv")
    features.to_csv(output_file, index=False)

    print(f"\nFeature matrix saved: {output_file}")
    print(f"Shape: {features.shape}")
    print(f"\nChurn rate: {features['churned'].mean():.2%}")
    print(f"\nSample:\n{features.head()}")
    print(f"\nNull counts:\n{features.isnull().sum()}")

    return features


if __name__ == "__main__":
    build_features()