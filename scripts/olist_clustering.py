import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set data path
data_path = Path('/hermes_workspace/Olist_e_commerce_project')

# 1. Load the data
print("Loading data...")
customers = pd.read_csv(data_path / 'olist_customers_dataset.csv')
orders = pd.read_csv(data_path / 'olist_orders_dataset.csv')
order_items = pd.read_csv(data_path / 'olist_order_items_dataset.csv')
order_payments = pd.read_csv(data_path / 'olist_order_payments_dataset.csv')
order_reviews = pd.read_csv(data_path / 'olist_order_reviews_dataset.csv')
products = pd.read_csv(data_path / 'olist_products_dataset.csv')
sellers = pd.read_csv(data_path / 'olist_sellers_dataset.csv')
product_category_translation = pd.read_csv(data_path / 'product_category_name_translation.csv')

# 2. Merge to create a comprehensive order view
print("Merging data...")

# Start with orders
df = orders.copy()

# Merge with customers to get customer details
df = df.merge(customers, on='customer_id', how='left')

# Merge with order_items to get items per order
# Note: one order can have multiple items, so we'll aggregate later
order_items_agg = order_items.groupby('order_id').agg(
    n_items=('order_item_id', 'count'),
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum'),
    avg_price=('price', 'mean'),
    avg_freight=('freight_value', 'mean'),
    # We can also get the first product_id, seller_id, etc. but for customer-level we might want to aggregate differently
    # For now, we'll keep the order-level aggregation and then aggregate to customer
).reset_index()

df = df.merge(order_items_agg, on='order_id', how='left')

# Merge with order_payments
payments_agg = order_payments.groupby('order_id').agg(
    n_payments=('payment_sequential', 'count'),
    total_payment_value=('payment_value', 'sum'),
    avg_payment_value=('payment_value', 'mean'),
    # We can also get the most common payment_type, but let's keep it simple for now
    # Alternatively, we can get the number of installments
    max_installments=('payment_installments', 'max'),
    avg_installments=('payment_installments', 'mean')
).reset_index()

df = df.merge(payments_agg, on='order_id', how='left')

# Merge with order_reviews
reviews_agg = order_reviews.groupby('order_id').agg(
    n_reviews=('review_id', 'count'),
    avg_review_score=('review_score', 'mean'),
    # We can also get the percentage of reviews with comments, but let's keep it simple
).reset_index()

df = df.merge(reviews_agg, on='order_id', how='left')

# 3. Now we have order-level features. Next, we aggregate to customer level.
print("Aggregating to customer level...")

# We'll group by customer_id and compute:
#   - Recency: days since last order (relative to the latest order in the dataset)
#   - Frequency: number of orders
#   - Monetary: total money spent (total_payment_value or total_price? We'll use total_payment_value)
#   - Other: average values, etc.

# First, convert date strings to datetime
date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Find the latest order date in the dataset
latest_date = df['order_purchase_timestamp'].max()
print(f"Latest order date in dataset: {latest_date}")

# Now aggregate by customer
customer_features = df.groupby('customer_id').agg(
    # Recency: days since last purchase
    recency=('order_purchase_timestamp', lambda x: (latest_date - x.max()).days),
    # Frequency: number of orders
    frequency=('order_id', 'nunique'),
    # Monetary: total payment value
    monetary=('total_payment_value', 'sum'),
    # Average monetary per order
    avg_monetary=('total_payment_value', 'mean'),
    # Average number of items per order
    avg_items_per_order=('n_items', 'mean'),
    # Average freight per order
    avg_freight_per_order=('total_freight', 'mean'),
    # Average payment installments
    avg_installments=('avg_installments', 'mean'),
    # Average review score
    avg_review_score=('avg_review_score', 'mean'),
    # Percentage of orders with reviews (if n_reviews > 0)
    pct_orders_with_review=('n_reviews', lambda x: (x > 0).sum() / len(x) * 100),
    # We can also add the diversity of products, sellers, etc. but let's keep it to these for now
).reset_index()

# 4. Merge in customer demographics (state, city, etc.)
customer_features = customer_features.merge(
    customers[['customer_id', 'customer_state', 'customer_city']], 
    on='customer_id', 
    how='left'
)

# 5. Handle missing values
# For numerical features, we'll fill with median (or mean) for now.
# For categorical, we'll fill with mode or create a missing category.

# Identify numerical and categorical columns
num_cols = customer_features.select_dtypes(include=[np.number]).columns.tolist()
# Exclude the customer_id from numerical columns for scaling
num_cols = [col for col in num_cols if col not in ['customer_id']]

cat_cols = customer_features.select_dtypes(include=['object']).columns.tolist()
# Exclude customer_id from categorical? We'll keep it for now but we won't encode it.
cat_cols = [col for col in cat_cols if col not in ['customer_id']]

print(f"Numerical columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

# Fill missing numerical with median
for col in num_cols:
    if customer_features[col].isnull().any():
        median_val = customer_features[col].median()
        customer_features[col].fillna(median_val, inplace=True)
        print(f"Filled missing {col} with median: {median_val}")

# Fill missing categorical with mode (or a constant)
for col in cat_cols:
    if customer_features[col].isnull().any():
        mode_val = customer_features[col].mode()[0] if not customer_features[col].mode().empty else 'missing'
        customer_features[col].fillna(mode_val, inplace=True)
        print(f"Filled missing {col} with mode: {mode_val}")

# 6. Encode categorical variables (state, city) - we'll use one-hot encoding for now
# But note: city has high cardinality, so we might want to target encode or just use state for simplicity.
# Let's use state only for clustering to avoid too many dimensions.
# We'll keep city for potential other uses but not for clustering.

# For clustering, we'll use:
#   - Numerical features (scaled)
#   - Categorical: state (one-hot encoded)

# Prepare the feature matrix for clustering
# We'll create a copy of customer_features for clustering
clustering_df = customer_features.copy()

# Select numerical features for scaling
scaler = StandardScaler()
num_features_scaled = scaler.fit_transform(clustering_df[num_cols])

# One-hot encode categorical features (state)
# We'll drop first to avoid multicollinearity
state_dummies = pd.get_dummies(clustering_df['customer_state'], prefix='state', drop_first=True)

# Combine numerical and categorical features
X = np.hstack([num_features_scaled, state_dummies.values])

print(f"Feature matrix shape: {X.shape}")

# 7. Run KMeans for a range of k and evaluate
print("Running KMeans for k=2 to 6...")
results = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Compute metrics
    sil = silhouette_score(X, cluster_labels)
    ch = calinski_harabasz_score(X, cluster_labels)
    db = davies_bouldin_score(X, cluster_labels)
    
    results.append({
        'k': k,
        'silhouette': sil,
        'calinski_harabasz': ch,
        'davies_bouldin': db,
        'labels': cluster_labels
    })
    
    print(f"k={k}: Silhouette={sil:.4f}, CH={ch:.1f}, DB={db:.3f}")

# Find the best k by silhouette score (higher is better)
best = max(results, key=lambda x: x['silhouette'])
best_k = best['k']
best_labels = best['labels']

print(f"\nBest k by silhouette: k={best_k} with silhouette={best['silhouette']:.4f}")

# 8. Add cluster labels to the customer_features dataframe
customer_features['cluster'] = best_labels

# 9. Analyze cluster characteristics
print("\nCluster sizes:")
print(customer_features['cluster'].value_counts().sort_index())

print("\nCluster means (numerical features):")
cluster_means = customer_features.groupby('cluster')[num_cols].mean()
print(cluster_means.round(2))

# Also look at categorical distribution (state)
print("\nCluster state distribution (top 2 states per cluster):")
for cluster in sorted(customer_features['cluster'].unique()):
    cluster_data = customer_features[customer_features['cluster'] == cluster]
    top_states = cluster_data['customer_state'].value_counts().head(2)
    print(f"Cluster {cluster}:")
    for state, count in top_states.items():
        print(f"  {state}: {count} ({count/len(cluster_data)*100:.1f}%)")

# 10. Save the customer features with clusters for later use
output_path = data_path / 'olist_customer_features_with_clusters.csv'
customer_features.to_csv(output_path, index=False)
print(f"\nSaved customer features with clusters to {output_path}")

# 11. Also save the feature matrix and model? We'll just save the CSV for now.

# 12. Generate a report
report_path = data_path / 'OLIST_CLUSTERING_REPORT.md'
with open(report_path, 'w') as f:
    f.write("# Olist E-commerce Customer Clustering Report\n\n")
    f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## Clustering Results\n\n")
    f.write(f"**Best number of clusters (k)**: {best_k}\n")
    f.write(f"**Silhouette Score**: {best['silhouette']:.4f}\n")
    f.write(f"**Calinski-Harabasz Score**: {best['calinski_harabasz']:.1f}\n")
    f.write(f"**Davies-Bouldin Score**: {best['davies_bouldin']:.3f}\n\n")
    f.write("### Cluster Sizes\n\n")
    f.write("| Cluster | Size | Percentage |\n")
    f.write("|---------|------|------------|\n")
    for cluster, size in customer_features['cluster'].value_counts().sort_index().items():
        pct = size / len(customer_features) * 100
        f.write(f"| {cluster} | {size} | {pct:.1f}% |\n")
    f.write("\n")
    f.write("### Cluster Characteristics (Numerical Features)\n\n")
    f.write("| Feature | ")
    f.write(" | ".join([f"Cluster {i}" for i in sorted(customer_features['cluster'].unique())]))
    f.write(" |\n")
    f.write("|---------|")
    f.write("|".join(["--------" for _ in customer_features['cluster'].unique()]))
    f.write("|\n")
    for feature in num_cols:
        f.write(f"| {feature} | ")
        f.write(" | ".join([f"{cluster_means.loc[cluster, feature]:.2f}" for cluster in sorted(customer_features['cluster'].unique())]))
        f.write(" |\n")
    f.write("\n")
    f.write("### Categorical Features (State) - Top 2 States per Cluster\n\n")
    for cluster in sorted(customer_features['cluster'].unique()):
        f.write(f"#### Cluster {cluster}\n\n")
        cluster_data = customer_features[customer_features['cluster'] == cluster]
        top_states = cluster_data['customer_state'].value_counts().head(2)
        f.write("| State | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        for state, count in top_states.items():
            pct = count / len(cluster_data) * 100
            f.write(f"| {state} | {count} | {pct:.1f}% |\n")
        f.write("\n")

print(f"Clustering report saved to {report_path}")

# 13. Also, let's create a simple scatter plot of two key features (e.g., recency vs monetary) colored by cluster
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(customer_features['recency'], customer_features['monetary'], 
                          c=customer_features['cluster'], cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Recency (days since last order)')
    plt.ylabel('Monetary (total payment value)')
    plt.title(f'Olist Customer Segmentation (k={best_k}): Recency vs Monetary')
    plt.grid(True, alpha=0.3)
    plot_path = data_path / 'olist_cluster_scatter_recency_monetary.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved scatter plot to {plot_path}")
except Exception as e:
    print(f"Could not create scatter plot: {e}")

print("\nClustering analysis complete.")