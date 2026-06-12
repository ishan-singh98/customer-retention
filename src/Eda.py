import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data path
DATA_PATH = r"C:\Users\Owner\.cache\kagglehub\datasets\psparks\instacart-market-basket-analysis\versions\1"
CHARTS_PATH = "outputs/charts"
os.makedirs(CHARTS_PATH, exist_ok=True)

# Load data
print("Loading data...")
orders = pd.read_csv(os.path.join(DATA_PATH, "orders.csv"))
order_products_prior = pd.read_csv(os.path.join(DATA_PATH, "order_products__prior.csv"))
products = pd.read_csv(os.path.join(DATA_PATH, "products.csv"))
departments = pd.read_csv(os.path.join(DATA_PATH, "departments.csv"))

print(f"Orders: {orders.shape}")
print(f"Order Products Prior: {order_products_prior.shape}")
print(f"Products: {products.shape}")

# Basic stats
print("\n--- Orders Sample ---")
print(orders.head())
print("\n--- Orders Info ---")
print(orders.info())
print("\n--- Days since prior order distribution ---")
print(orders["days_since_prior_order"].describe())

# Plot 1: Order frequency per user
order_counts = orders.groupby("user_id")["order_number"].max().reset_index()
order_counts.columns = ["user_id", "total_orders"]

plt.figure(figsize=(10, 5))
sns.histplot(order_counts["total_orders"], bins=50, kde=True, color="steelblue")
plt.title("Distribution of Total Orders per Customer")
plt.xlabel("Total Orders")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "order_frequency_distribution.png"))
plt.close()
print("Saved: order_frequency_distribution.png")

# Plot 2: Days since prior order
days_data = orders[orders["days_since_prior_order"].notna()]

plt.figure(figsize=(10, 5))
sns.histplot(days_data["days_since_prior_order"], bins=30, kde=True, color="coral")
plt.title("Distribution of Days Since Prior Order")
plt.xlabel("Days")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "days_since_prior_order.png"))
plt.close()
print("Saved: days_since_prior_order.png")

# Plot 3: Orders by day of week
plt.figure(figsize=(8, 5))
orders["order_dow"].value_counts().sort_index().plot(kind="bar", color="mediumseagreen")
plt.title("Orders by Day of Week (0=Sunday)")
plt.xlabel("Day of Week")
plt.ylabel("Order Count")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "orders_by_dow.png"))
plt.close()
print("Saved: orders_by_dow.png")

# Plot 4: Orders by hour of day
plt.figure(figsize=(10, 5))
orders["order_hour_of_day"].value_counts().sort_index().plot(kind="bar", color="mediumpurple")
plt.title("Orders by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Order Count")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "orders_by_hour.png"))
plt.close()
print("Saved: orders_by_hour.png")

# Plot 5: Basket size distribution
basket_sizes = order_products_prior.groupby("order_id")["product_id"].count().reset_index()
basket_sizes.columns = ["order_id", "basket_size"]

plt.figure(figsize=(10, 5))
sns.histplot(basket_sizes["basket_size"], bins=50, kde=True, color="darkorange")
plt.title("Distribution of Basket Size")
plt.xlabel("Number of Products per Order")
plt.ylabel("Count")
plt.xlim(0, 60)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "basket_size_distribution.png"))
plt.close()
print("Saved: basket_size_distribution.png")

# Churn label preview
print("\n--- Churn Label Preview ---")
print("eval_set distribution:")
print(orders["eval_set"].value_counts())
# Customers in 'prior' only with no 'train' or 'test' record = potential churners
user_eval = orders.groupby("user_id")["eval_set"].apply(set).reset_index()
churned = user_eval[user_eval["eval_set"] == {"prior"}]
print(f"\nCustomers with only prior orders (potential churners): {len(churned)}")
print(f"Total unique customers: {orders['user_id'].nunique()}")
print(f"Churn rate estimate: {len(churned)/orders['user_id'].nunique():.2%}")

print("\nEDA complete. Charts saved to outputs/charts/")