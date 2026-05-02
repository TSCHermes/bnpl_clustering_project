import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Set data path
data_path = Path('/hermes_workspace/Olist_e_commerce_project')

print("Step 1: Loading customers and taking a small sample...")
customers = pd.read_csv(data_path / 'olist_customers_dataset.csv', 
                        usecols=['customer_id', 'customer_unique_id', 'customer_state', 'customer_city'])
# Sample 0.5% of customers for faster processing but still meaningful
sample_frac = 0.005
sampled_customers = customers.sample(frac=sample_frac, random_state=42)
print(f"Sampled {len(sampled_customers)} customers out of {len(customers)} ({sample_frac*100}%)")

# Get the sampled customer IDs
sampled_customer_ids = set(sampled_customers['customer_unique_id'])

print("\nStep 2: Loading and filtering orders...")
orders = pd.read_csv(data_path / 'olist_orders_dataset.csv', 
                     usecols=['order_id', 'customer_id', 'order_purchase_timestamp'])
# Merge with customers to get customer_unique_id for filtering
orders_with_customer = pd.merge(orders, customers[['customer_id', 'customer_unique_id']], on='customer_id')
# Filter to sampled customers
orders_sampled = orders_with_customer[orders_with_customer['customer_unique_id'].isin(sampled_customer_ids)]
print(f"Sampled {len(orders_sampled)} orders out of {len(orders)}")
print(f"Columns in orders_sampled: {orders_sampled.columns.tolist()}")

# Free memory
del orders, orders_with_customer

print("\nStep 3: Loading and filtering order items...")
order_items = pd.read_csv(data_path / 'olist_order_items_dataset.csv', 
                          usecols=['order_id', 'order_item_id', 'product_id', 'seller_id', 'price', 'freight_value'])
order_items_sampled = order_items[order_items['order_id'].isin(orders_sampled['order_id'])]
print(f"Sampled {len(order_items_sampled)} order items out of {len(order_items)}")
del order_items

print("\nStep 4: Loading and filtering order payments...")
order_payments = pd.read_csv(data_path / 'olist_order_payments_dataset.csv', 
                             usecols=['order_id', 'payment_value', 'payment_installments'])
order_payments_sampled = order_payments[order_payments['order_id'].isin(orders_sampled['order_id'])]
print(f"Sampled {len(order_payments_sampled)} order payments out of {len(order_payments)}")
del order_payments

print("\nStep 5: Loading and filtering order reviews...")
order_reviews = pd.read_csv(data_path / 'olist_order_reviews_dataset.csv', 
                            usecols=['order_id', 'review_score', 'review_comment_title', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp'])
order_reviews_sampled = order_reviews[order_reviews['order_id'].isin(orders_sampled['order_id'])]
print(f"Sampled {len(order_reviews_sampled)} order reviews out of {len(order_reviews)}")
del order_reviews

print("\nStep 6: Converting date columns...")
orders_sampled['order_purchase_timestamp'] = pd.to_datetime(orders_sampled['order_purchase_timestamp'])
order_reviews_sampled['review_creation_date'] = pd.to_datetime(order_reviews_sampled['review_creation_date'])
order_reviews_sampled['review_answer_timestamp'] = pd.to_datetime(order_reviews_sampled['review_answer_timestamp'])

print("\nStep 7: Merging data to create customer-level features...")

# Start with orders and customers (we already have customer_unique_id in orders_sampled)
# But we need the customer_state and customer_city from customers
orders_customers = pd.merge(orders_sampled, 
                            customers[['customer_unique_id', 'customer_state', 'customer_city']], 
                            on='customer_unique_id')
print(f"After merging with customers: {len(orders_customers)} rows")

# Add order items
orders_items = pd.merge(orders_customers, 
                        order_items_sampled[['order_id', 'order_item_id', 'product_id', 'seller_id', 'price', 'freight_value']], 
                        on='order_id')
print(f"After merging order items: {len(orders_items)} rows")

# Load products and product category translation (small tables)
products = pd.read_csv(data_path / 'olist_products_dataset.csv', 
                       usecols=['product_id', 'product_category_name'])
product_category_name_translation = pd.read_csv(data_path / 'product_category_name_translation.csv', 
                                                usecols=['product_category_name', 'product_category_name_english'])
products_with_category = pd.merge(products, product_category_name_translation[['product_category_name', 'product_category_name_english']], on='product_category_name', how='left')
orders_items_products = pd.merge(orders_items, products_with_category[['product_id', 'product_category_name_english']], on='product_id')
print(f"After merging product info: {len(orders_items_products)} rows")

# Load sellers (small table)
sellers = pd.read_csv(data_path / 'olist_sellers_dataset.csv', 
                      usecols=['seller_id', 'seller_state'])
orders_items_products_sellers = pd.merge(orders_items_products, sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
print(f"After merging seller info: {len(orders_items_products_sellers)} rows")

# Add payments
orders_items_products_sellers_payments = pd.merge(orders_items_products_sellers, order_payments_sampled, on='order_id')
print(f"After merging payments: {len(orders_items_products_sellers_payments)} rows")

# Add reviews
orders_items_products_sellers_payments_reviews = pd.merge(orders_items_products_sellers_payments, order_reviews_sampled, on='order_id', how='left')
print(f"After merging reviews: {len(orders_items_products_sellers_payments_reviews)} rows")
print(f"Columns in final merged: {orders_items_products_sellers_payments_reviews.columns.tolist()}")

print("\nStep 8: Aggregating to customer level...")
# Compute the latest date in the dataset for recency calculation
latest_order_date = orders_items_products_sellers_payments_reviews['order_purchase_timestamp'].max()
print(f"Latest order date in dataset: {latest_order_date}")

customer_features = orders_items_products_sellers_payments_reviews.groupby('customer_unique_id').agg(
    # RFM features
    recency=('order_purchase_timestamp', lambda x: (latest_order_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('payment_value', 'sum'),
    # Additional behavioral features
    avg_monetary=('payment_value', 'mean'),
    avg_items_per_order=('order_item_id', 'mean'),
    avg_freight_per_order=('freight_value', 'mean'),
    avg_installments=('payment_installments', 'mean'),
    avg_review_score=('review_score', 'mean'),  # Use review_score directly
    pct_orders_with_review=('review_score', lambda x: x.notna().mean()),
    # Categorical features
    customer_state=('customer_state', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
    customer_city=('customer_city', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
    # Product diversity
    unique_product_categories=('product_category_name_english', 'nunique'),
    unique_sellers=('seller_id', 'nunique'),
).reset_index()

print(f"Aggregated to {len(customer_features)} unique customers")
print(f"Customer features columns: {customer_features.columns.tolist()}")

print("\nStep 9: Handling missing values...")
# Fill numerical missing values with median
num_cols = ['avg_monetary', 'avg_items_per_order', 'avg_freight_per_order', 'avg_installments', 'avg_review_score', 'pct_orders_with_review']
for col in num_cols:
    if col in customer_features.columns:
        median_val = customer_features[col].median()
        customer_features[col] = customer_features[col].fillna(median_val)
        print(f"Filled missing {col} with median: {median_val:.2f}")

print("\nStep 10: Preparing features for clustering...")
# Separate features
numerical_cols = ['recency', 'frequency', 'monetary', 'avg_monetary', 'avg_items_per_order', 'avg_freight_per_order', 'avg_installments', 'avg_review_score', 'pct_orders_with_review', 'unique_product_categories', 'unique_sellers']
categorical_cols = ['customer_state', 'customer_city']

# Ensure all numerical columns exist
numerical_cols = [col for col in numerical_cols if col in customer_features.columns]

# Extract features
X_num = customer_features[numerical_cols].copy()
X_cat = pd.get_dummies(customer_features[categorical_cols], drop_first=True)

# Combine features
X = pd.concat([X_num, X_cat], axis=1)

# Handle any remaining NaN or inf values
print(f"Feature matrix shape before cleaning: {X.shape}")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
print(f"Feature matrix shape after cleaning: {X.shape}")

print("\nStep 11: Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nStep 12: Applying PCA for dimensionality reduction (to 95% variance)...")
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Original features: {X_scaled.shape[1]}")
print(f"PCA components: {X_pca.shape[1]}")
print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

print("\nStep 13: Clustering with MiniBatchKMeans...")
results = []
for k in range(2, 6):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100, max_iter=20)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    # Compute metrics
    sil = silhouette_score(X_pca, cluster_labels)
    ch = calinski_harabasz_score(X_pca, cluster_labels)
    db = davies_bouldin_score(X_pca, cluster_labels)
    
    results.append({
        'k': k,
        'silhouette': sil,
        'calinski_harabasz': ch,
        'davies_bouldin': db,
        'labels': cluster_labels
    })
    
    print(f"k={k}: Silhouette={sil:.4f}, CH={ch:.1f}, DB={db:.3f}")

# Select best k based on silhouette score
best_idx = np.argmax([r['silhouette'] for r in results])
best = results[best_idx]
best_k = best['k']

print(f"\nBest number of clusters: {best_k}")
print(f"Best Silhouette Score: {best['silhouette']:.4f}")

# Add cluster labels to customer features
customer_features['cluster'] = best['labels']

# Save results
output_path = data_path / f'olist_customer_clusters_sampled_{int(sample_frac*100)}pct.csv'
customer_features.to_csv(output_path, index=False)
print(f"\nSaved customer features with clusters ({sample_frac*100}% sample) to {output_path}")

# Generate report
report_path = data_path / f'OLIST_CLUSTERING_REPORT_SAMPLED_{int(sample_frac*100)}PCT.md'
with open(report_path, 'w') as f:
    f.write(f"# Olist E-commerce Customer Clustering Report ({sample_frac*100}% Sample)\n\n")
    f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Note**: This analysis is based on a {sample_frac*100}% random sample of customers for quick testing.\n\n")
    
    f.write("## Clustering Results\n\n")
    f.write(f"**Best number of clusters (k)**: {best_k}\n\n")
    f.write(f"**Silhouette Score**: {best['silhouette']:.4f}\n\n")
    f.write(f"**Calinski-Harabasz Score**: {best['calinski_harabasz']:.1f}\n\n")
    f.write(f"**Davies-Bouldin Score**: {best['davies_bouldin']:.3f}\n\n")
    
    f.write("### PCA Information\n\n")
    f.write(f"- Original features: {X_scaled.shape[1]}\n\n")
    f.write(f"- PCA components: {X_pca.shape[1]}\n\n")
    f.write(f"- Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}\n\n")
    
    f.write("### Cluster Sizes\n\n")
    f.write("| Cluster | Size | Percentage |\n")
    f.write("|---------|------|------------|\n")
    cluster_sizes = customer_features['cluster'].value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
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
    
    cluster_means = customer_features.groupby('cluster')[numerical_cols].mean()
    for feature in numerical_cols:
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
    
    f.write("### Categorical Features (City) - Top 2 Cities per Cluster\n\n")
    for cluster in sorted(customer_features['cluster'].unique()):
        f.write(f"#### Cluster {cluster} (City)\n\n")
        cluster_data = customer_features[customer_features['cluster'] == cluster]
        top_cities = cluster_data['customer_city'].value_counts().head(2)
        f.write("| City | Count | Percentage |\n")
        f.write("|------|-------|------------|\n")
        for city, count in top_cities.items():
            pct = count / len(cluster_data) * 100
            f.write(f"| {city} | {count} | {pct:.1f}% |\n")
        f.write("\n")

print(f"Clustering report saved to {report_path}")

# Create a simple scatter plot of two key features (using original features for interpretability)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(customer_features['recency'], customer_features['monetary'], 
                          c=customer_features['cluster'], cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Recency (days since last order)')
    plt.ylabel('Monetary (total payment value)')
    plt.title(f'Olist Customer Segmentation (k={best_k}): Recency vs Monetary ({sample_frac*100}% Sample)')
    plt.grid(True, alpha=0.3)
    plot_path = data_path / f'olist_cluster_scatter_recency_monetary_sampled_{int(sample_frac*100)}pct.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved scatter plot to {plot_path}")
except Exception as e:
    print(f"Could not create scatter plot: {e}")

print("\nClustering analysis complete.")