# Olist E-commerce Customer Clustering Report (0.5% Sample)

Generated on: 2026-05-02 12:08:45

**Note**: This analysis is based on a 0.5% random sample of customers for quick testing.

## Clustering Results

**Best number of clusters (k)**: 3

**Silhouette Score**: -0.1201

**Calinski-Harabasz Score**: 2.5

**Davies-Bouldin Score**: 3.850

### PCA Information

- Original features: 272

- PCA components: 228

- Explained variance ratio: 0.9536

### Cluster Sizes

| Cluster | Size | Percentage |
|---------|------|------------|
| 0 | 1 | 0.2% |
| 1 | 411 | 83.2% |
| 2 | 82 | 16.6% |

### Cluster Characteristics (Numerical Features)

| Feature | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|--------|--------|--------|
| recency | 249.00 | 242.45 | 253.24 |
| frequency | 1.00 | 1.09 | 1.07 |
| monetary | 44.69 | 430.25 | 185.14 |
| avg_monetary | 44.69 | 173.60 | 119.47 |
| avg_items_per_order | 1.00 | 1.07 | 1.07 |
| avg_freight_per_order | 16.79 | 20.42 | 14.96 |
| avg_installments | 4.00 | 3.10 | 2.51 |
| avg_review_score | 5.00 | 4.20 | 4.08 |
| pct_orders_with_review | 1.00 | 0.99 | 1.00 |
| unique_product_categories | 0.00 | 1.07 | 1.01 |
| unique_sellers | 1.00 | 1.09 | 1.02 |

### Categorical Features (State) - Top 2 States per Cluster

#### Cluster 0

| State | Count | Percentage |
|-------|-------|------------|
| PR | 1 | 100.0% |

#### Cluster 1

| State | Count | Percentage |
|-------|-------|------------|
| SP | 143 | 34.8% |
| MG | 56 | 13.6% |

#### Cluster 2

| State | Count | Percentage |
|-------|-------|------------|
| SP | 82 | 100.0% |

### Categorical Features (City) - Top 2 Cities per Cluster

#### Cluster 0 (City)

| City | Count | Percentage |
|------|-------|------------|
| apucarana | 1 | 100.0% |

#### Cluster 1 (City)

| City | Count | Percentage |
|------|-------|------------|
| rio de janeiro | 27 | 6.6% |
| belo horizonte | 15 | 3.6% |

#### Cluster 2 (City)

| City | Count | Percentage |
|------|-------|------------|
| sao paulo | 80 | 97.6% |
| matao | 1 | 1.2% |

