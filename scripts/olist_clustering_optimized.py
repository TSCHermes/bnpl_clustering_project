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

# 1. Load the data with only necessary columns
print("Loading data...")
customers = pd.read_csv(data_path / 'olist_customers_dataset.csv', 
                        usecols=['customer_id', 'customer_unique_id', 'customer_state', 'customer_city'])
orders = pd.read_csv(data_path / 'olist_orders_dataset.csv', 
                     usecols=['order_id', 'customer_id', 'order_purchase_timestamp'])
order_items = pd.read_csv(data_path / 'olist_order_items_dataset.csv', 
                          usecols=['order_id', 'order_item_id', 'product_id', 'seller_id', 'price', 'freight_value'])
order_payments = pd.read_csv(data_path / 'olist_order_payments_dataset.csv', 
                             usecols=['order_id', 'payment_value', 'payment_installments'])
order_reviews = pd.read_csv(data_path / 'olist_order_reviews_dataset.csv', 
                            usecols=['order_id', 'review_score', 'review_comment_title', 'review_comment_message'])
sellers = pd.read_csv(data_path / 'olist_sellers_dataset.csv', 
                      usecols=['seller_id', 'seller_state'])
products = pd.read_csv(data_path / 'olist_products_dataset.csv', 
                       usecols=['product_id', 'product_category_name'])
product_category_name_translation = pd.read_csv(data_path / 'product_category_name_translation.csv', 
                                                usecols=['product_category_name', 'product_category_name_english'])

# Convert date columns
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
order_reviews['review_creation_date'] = pd.to_datetime(order_reviews['review_creation_date'])
order_reviews['review_answer_timestamp'] = pd.to_datetime(order_reviews['review_answer_timestamp'])

# 2. Merge data to create customer-level features
print("Merging data...")

# Start with orders and customers
orders_customers = pd.merge(orders, customers[['customer_id', 'customer_unique_id', 'customer_state', 'customer_city']], on='customer_id')

# Add order items
orders_items = pd.merge(orders_customers, order_items[['order_id', 'order_item_id', 'product_id', 'seller_id', 'price', 'freight_value']], on='order_id')

# Add product information
products_with_category = pd.merge(products, product_category_name_translation[['product_category_name', 'product_category_name_english']], on='product_category_name', how='left')
orders_items_products = pd.merge(orders_items, products_with_category[['product_id', 'product_category_name_english']], on='product_id')

# Add seller information
orders_items_products_sellers = pd.merge(orders_items_products, sellers[['seller_id', 'seller_state']], on='seller_id', how='left')

# Add payments
order_payments_agg = order_payments.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_installments': 'mean'
}).reset_index()
orders_items_products_sellers_payments = pd.merge(orders_items_products_sellers, order_payments_agg, on='order_id')

# Add reviews
order_reviews_agg = order_reviews.groupby('order_id').agg({
    'review_score': 'mean',
    'review_comment_title': 'count',
    'review_comment_message': 'count'
}).reset_index()
order_reviews_agg.columns = ['order_id', 'avg_review_score', 'comment_title_count', 'comment_message_count']
orders_items_products_sellers_payments_reviews = pd.merge(orders_items_products_sellers_payments, order_reviews_agg, on='order_id', how='left')

# 3. Aggregate to customer level
print("Aggregating to customer level...")
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
    avg_review_score=('avg_review_score', 'mean'),
    pct_orders_with_review=('avg_review_score', lambda x: x.notna().mean()),
    # Categorical features
    customer_state=('customer_state', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
    customer_city=('customer_city', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
    # Product diversity
    unique_product_categories=('product_category_name_english', 'nunique'),
    unique_sellers=('seller_id', 'nunique'),
).reset_index()

# 4. Handle missing values
print("Handling missing values...")
# Fill numerical missing values with median
num_cols = ['avg_monetary', 'avg_items_per_order', 'avg_freight_per_order', 'avg_installments', 'avg_review_score', 'pct_orders_with_review']
for col in num_cols:
    if col in customer_features.columns:
        median_val = customer_features[col].median()
        customer_features[col] = customer_features[col].fillna(median_val)
        print(f"Filled missing {col} with median: {median_val:.2f}")

# 5. Take a 1% sample for quick testing
print("Taking a 1% sample for quick testing...")
customer_features_sample = customer_features.sample(frac=0.01, random_state=42)
print(f"Sample size: {len(customer_features_sample)} customers")

# 6. Prepare features for clustering
print("Preparing features for clustering...")
# Separate features
numerical_cols = ['recency', 'frequency', 'monetary', 'avg_monetary', 'avg_items_per_order', 'avg_freight_per_order', 'avg_installments', 'avg_review_score', 'pct_orders_with_review', 'unique_product_categories', 'unique_sellers']
categorical_cols = ['customer_state', 'customer_city']

# Ensure all numerical columns exist
numerical_cols = [col for col in numerical_cols if col in customer_features_sample.columns]

# Extract features
X_num = customer_features_sample[numerical_cols].copy()
X_cat = pd.get_dummies(customer_features_sample[categorical_cols], drop_first=True)

# Combine features
X = pd.concat([X_num, X_cat], axis=1)

# Handle any remaining NaN or inf values
print(f"Feature matrix shape before cleaning: {X.shape}")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
print(f"Feature matrix shape after cleaning: {X.shape}")

# 7. Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Apply PCA for dimensionality reduction (to 95% variance)
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Original features: {X_scaled.shape[1]}")
print(f"PCA components: {X_pca.shape[1]}")
print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

# 9. Clustering with MiniBatchKMeans (more efficient)
print("Running MiniBatchKMeans for k=2 to 5...")
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

# 10. Select best k based on silhouette score
best_idx = np.argmax([r['silhouette'] for r in results])
best = results[best_idx]
best_k = best['k']

print(f"\nBest number of clusters: {best_k}")
print(f"Best Silhouette Score: {best['silhouette']:.4f}")

# 11. Add cluster labels to customer features (sample)
customer_features_sample['cluster'] = best['labels']

# 12. Save results (sample)
output_path = data_path / 'olist_customer_clusters_optimized_1pct.csv'
customer_features_sample.to_csv(output_path, index=False)
print(f"\nSaved customer features with clusters (1% sample) to {output_path}")

# 13. Generate report
report_path = data_path / 'OLIST_CLUSTERING_REPORT_OPTIMIZED.md'
with open(report_path, 'w') as f:
    f.write("# Olist E-commerce Customer Clustering Report (Optimized 1% Sample)\n\n")
    f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("**Note**: This analysis is based on a 1% random sample of customers for quick testing.\n\n")
    
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
    cluster_sizes = customer_features_sample['cluster'].value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
        pct = size / len(customer_features_sample) * 100
        f.write(f"| {cluster} | {size} | {pct:.1f}% |\n")
    f.write("\n")
    
    f.write("### Cluster Characteristics (Numerical Features)\n\n")
    f.write("| Feature | ")
    f.write(" | ".join([f"Cluster {i}" for i in sorted(customer_features_sample['cluster'].unique())]))
    f.write(" |\n")
    f.write("|---------|")
    f.write("|".join(["--------" for _ in customer_features_sample['cluster'].unique()]))
    f.write("|\n")
    
    cluster_means = customer_features_sample.groupby('cluster')[numerical_cols].mean()
    for feature in numerical_cols:
        f.write(f"| {feature} | ")
        f.write(" | ".join([f"{cluster_means.loc[cluster, feature]:.2f}" for cluster in sorted(customer_features_sample['cluster'].unique())]))
        f.write(" |\n")
    f.write("\n")
    
    f.write("### Categorical Features (State) - Top 2 States per Cluster\n\n")
    for cluster in sorted(customer_features_sample['cluster'].unique()):
        f.write(f"#### Cluster {cluster}\n\n")
        cluster_data = customer_features_sample[customer_features_sample['cluster'] == cluster]
        top_states = cluster_data['customer_state'].value_counts().head(2)
        f.write("| State | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        for state, count in top_states.items():
            pct = count / len(cluster_data) * 100
            f.write(f"| {state} | {count} | {pct:.1f}% |\n")
        f.write("\n")
    
    f.write("### Categorical Features (City) - Top 2 Cities per Cluster\n\n")
    for cluster in sorted(customer_features_sample['cluster'].unique()):
        f.write(f"#### Cluster {cluster} (City)\n\n")
        cluster_data = customer_features_sample[customer_features_sample['cluster'] == cluster]
        top_cities = cluster_data['customer_city'].value_counts().head(2)
        f.write("| City | Count | Percentage |\n")
        f.write("|------|-------|------------|\n")
        for city, count in top_cities.items():
            pct = count / len(cluster_data) * 100
            f.write(f"| {city} | {count} | {pct:.1f}% |\n")
        f.write("\n")

print(f"Clustering report saved to {report_path}")

# 14. Create a simple scatter plot of two key features (using original features for interpretability)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(customer_features_sample['recency'], customer_features_sample['monetary'], 
                          c=customer_features_sample['cluster'], cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Recency (days since last order)')
    plt.ylabel('Monetary (total payment value)')
    plt.title(f'Olist Customer Segmentation (k={best_k}): Recency vs Monetary (Optimized 1% Sample)')
    plt.grid(True, alpha=0.3)
    plot_path = data_path / 'olist_cluster_scatter_recency_monetary_optimized.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved scatter plot to {plot_path}")
except Exception as e:
    print(f"Could not create scatter plot: {e}")

print("\nClustering analysis complete.")