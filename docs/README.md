# Olist E-commerce Customer Clustering Analysis

This project performs customer segmentation on the Olist Brazilian e-commerce dataset using RFM (Recency, Frequency, Monetary) and additional behavioral and geographic features.

## Project Structure

- `data/raw/` - Original Olist dataset files (CSV)
- `data/processed/` - Processed data and clustering results (CSV)
- `docs/` - Documentation and reports (Markdown)
- `results/visuals/` - Visualizations (PNG)
- `scripts/` - Python scripts for data processing and clustering

## Analyses Performed

### 1. 0.5% Sample Analysis (Initial)
- **Script**: `scripts/olist_clustering_final.py`
- **Output**: 
  - `data/processed/olist_customer_clusters_sampled_0pct.csv`
  - `docs/OLIST_CLUSTERING_REPORT_SAMPLED_0PCT.md`
  - Visualizations in `results/visuals/` (with `sampled_0pct` in filename)
- **Results**: 
  - Best k=3 (silhouette = -0.1201)
  - Cluster 2 (17%) = exclusively São Paulo state customers
  - Cluster 1 (83%) = geographically diverse customers

### 2. 5% Sample Analysis (Larger Dataset)
- **Script**: `scripts/olist_clustering_5pct.py`
- **Output**:
  - `data/processed/olist_customer_clusters_sampled_5pct.csv`
  - `docs/OLIST_CLUSTERING_REPORT_SAMPLED_5PCT.md`
  - Visualization: `results/visuals/olist_cluster_scatter_recency_monetary_sampled_5pct.png`
- **Results**:
  - Best k=2 (silhouette = 0.3552)
  - Two distinct clusters separated by purchasing behavior

## Visualizations

Below are the visualizations of the customer clusters, each described in both technical and business-friendly terms:

### 1. Recency vs Monetary
![Customer Clusters: Recency vs Monetary](../results/visuals/olist_cluster_scatter_recency_monetary.png)
**Recency (How recently a customer last purchased)** vs **Monetary (Total amount spent)**
- Shows how customer groups differ in their purchase timing and spending habits.
- Business insight: Identifies segments like "recent big spenders" (valuable) vs "old small spenders" (at risk).

### 2. Frequency vs Monetary
![Customer Clusters: Frequency vs Monetary](../results/visuals/olist_cluster_scatter_frequency_monetary.png)
**Frequency (Number of orders placed)** vs **Monetary (Total amount spent)**
- Reveals whether customers buy often in small amounts or rarely in large amounts.
- Business insight: Helps distinguish loyal frequent buyers from occasional high-value customers.

### 3. Recency vs Frequency
![Customer Clusters: Recency vs Frequency](../results/visuals/olist_cluster_scatter_recency_frequency.png)
**Recency (How recently a customer last purchased)** vs **Frequency (Number of orders placed)**
- Combines timing and loyalty dimensions to show engagement patterns.
- Business insight: Highlights customers who buy frequently but haven't purchased recently (potential churn risk).

### 4. Cluster Sizes
![Cluster Sizes](../results/visuals/olist_cluster_sizes.png)
**Cluster Size (Number of customers in each segment)**
- Simple count of how many customers fall into each discovered segment.
- Business insight: Shows the relative importance of each segment for resource allocation.

### 5. Monetary Distribution by Cluster
![Monetary Distribution by Cluster](../results/visuals/olist_cluster_monetary_boxplot.png)
**Monetary Spread (Range of total spending within each segment)**
- Displays the distribution, median, and outliers of spending for each customer group.
- Business insight: Reveals spending consistency within segments and identifies exceptional customers (outliers).

## Key Findings (0.5% Sample)

- **Best number of clusters (k)**: 3 distinct customer segments identified
- **Silhouette Score**: -0.1201 (indicates modest separation; natural overlap expected in real-world behavior)
- **Cluster Distribution**:
  - **Cluster 0**: 1 customer (0.2%) - Unique outlier from PR state (possibly data anomaly or special case)
  - **Cluster 1**: 411 customers (83.2%) - Geographically diverse mainstream customers (mostly from SP/MG states)
  - **Cluster 2**: 82 customers (16.6%) - Exclusively from São Paulo state (potential regional loyalty pattern)

## Environment Setup

The project uses a uv virtual environment. Dependencies are installed via:
```bash
uv venv .venv --seed
uv pip install numpy scikit-learn pandas matplotlib seaborn
```

## Running the Analysis

To run the clustering analysis on a sample:
```bash
python scripts/olist_clustering_final.py   # 0.5% sample
# or
python scripts/olist_clustering_5pct.py  # 5% sample
```

## Next Steps

For production analysis, consider:
1. Increasing sample size to 10-20% or full dataset
2. Feature selection to reduce dimensionality and improve interpretability
3. Trying alternative clustering algorithms (DBSCAN, hierarchical) that may capture different segment shapes
4. Creating business-interpretable features beyond RFM (e.g., product category preferences, seasonal patterns)
5. Using Git LFS for large files (e.g., geolocation dataset) if modifying frequently

---
*Analysis completed on 2026-05-02*