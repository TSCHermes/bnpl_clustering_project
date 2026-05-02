import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load clustered data
df = pd.read_csv('/hermes_workspace/Olist_e_commerce_project/olist_customer_clusters_sampled_0pct.csv')

# Ensure output directory
output_dir = '/hermes_workspace/Olist_e_commerce_project'
os.makedirs(output_dir, exist_ok=True)

# 1. Scatter Recency vs Monetary
plt.figure()
scatter = plt.scatter(df['recency'], df['monetary'], c=df['cluster'], cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Recency (days since last order)')
plt.ylabel('Monetary (total payment value)')
plt.title('Customer Clusters: Recency vs Monetary')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'olist_cluster_scatter_recency_monetary.png'), dpi=150)
plt.close()
print('Saved: olist_cluster_scatter_recency_monetary.png')

# 2. Scatter Frequency vs Monetary
plt.figure()
scatter = plt.scatter(df['frequency'], df['monetary'], c=df['cluster'], cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Frequency (number of orders)')
plt.ylabel('Monetary (total payment value)')
plt.title('Customer Clusters: Frequency vs Monetary')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'olist_cluster_scatter_frequency_monetary.png'), dpi=150)
plt.close()
print('Saved: olist_cluster_scatter_frequency_monetary.png')

# 3. Scatter Recency vs Frequency
plt.figure()
scatter = plt.scatter(df['recency'], df['frequency'], c=df['cluster'], cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Recency (days since last order)')
plt.ylabel('Frequency (number of orders)')
plt.title('Customer Clusters: Recency vs Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'olist_cluster_scatter_recency_frequency.png'), dpi=150)
plt.close()
print('Saved: olist_cluster_scatter_recency_frequency.png')

# 4. Bar chart of cluster sizes
plt.figure()
cluster_counts = df['cluster'].value_counts().sort_index()
bars = plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue', edgecolor='navy')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Cluster Sizes')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'olist_cluster_sizes.png'), dpi=150)
plt.close()
print('Saved: olist_cluster_sizes.png')

# 5. Boxplot of Monetary per cluster
plt.figure()
sns.boxplot(x='cluster', y='monetary', data=df, palette='viridis')
plt.xlabel('Cluster')
plt.ylabel('Monetary (total payment value)')
plt.title('Monetary Distribution by Cluster')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'olist_cluster_monetary_boxplot.png'), dpi=150)
plt.close()
print('Saved: olist_cluster_monetary_boxplot.png')

print('All 5 plots generated.')