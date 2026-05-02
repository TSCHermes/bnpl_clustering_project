# Clustering Analysis Plan for Olist E-commerce Dataset

## Objective
Perform customer segmentation using clustering techniques to identify distinct customer groups for targeted marketing and product strategies.

## Prerequisites
- Completed EDA (OLIST_EDA_REPORT.md)
- Feature engineering script ready (olist_clustering_fixed.py)
- Python environment with pandas, numpy, scikit-learn, matplotlib, seaborn

## Step-by-Step Plan

### 1. Environment Setup
- [ ] Ensure virtual environment is activated with required packages
- [ ] Verify no conflicting files (e.g., numpy.py, pandas.py) in project directory
- [ ] Install missing packages if needed: `uv pip install pandas numpy matplotlib seaborn scikit-learn`

### 2. Data Integration & Feature Engineering
- [ ] Execute the feature engineering script to create customer-level dataset
  - Merge orders, order_items, order_payments, order_reviews, customers, sellers, products, geolocation
  - Calculate RFM features:
    * Recency: days since last purchase (relative to max order date in dataset)
    * Frequency: count of orders per customer
    * Monetary: sum of payment_value per customer
  - Calculate additional features:
    * Average items per order
    * Average freight value
    * Average payment installments
    * Percentage of orders with reviews
    * Average review score
    * Customer state (categorical)
    * Optional: product category diversity, seller count, etc.
- [ ] Handle missing values:
  * Numerical features: fill with median
  * Categorical features (state): fill with mode
- [ ] Encode categorical variables:
  * State: one-hot encoding (or target encoding if high cardinality becomes issue)
- [ ] Feature scaling:
  * Standardize numerical features (zero mean, unit variance)
  * Leave one-hot encoded features as is (0/1)

### 3. Clustering Model Selection & Training
- [ ] Determine optimal number of clusters (k) using elbow method and silhouette analysis
  * Test k from 2 to 6 (or higher if justified)
  * For each k:
    - Train KMeans model (random_state=42 for reproducibility)
    - Calculate inertia (within-cluster sum of squares)
    - Calculate silhouette score
    - Calculate Calinski-Harabasz index
    - Calculate Davies-Bouldin index
- [ ] Select optimal k based on:
  * Peak silhouette score
  * Elbow point in inertia curve
  * Business interpretability of clusters
- [ ] Train final KMeans model with selected k

### 4. Cluster Analysis & Interpretation
- [ ] Assign cluster labels to each customer
- [ ] Calculate cluster profiles:
  * Mean/median of each feature per cluster
  * Size (percentage of total customers) per cluster
  * Distinguishing characteristics (features significantly different from overall mean)
- [ ] Visualize clusters:
  * Scatter plots of key feature pairs (e.g., Recency vs Monetary, Frequency vs Monetary)
  * Bar charts showing average feature values per cluster
  * Pie chart showing cluster size distribution
- [ ] Business interpretation:
  * Label clusters with descriptive names (e.g., "High-Value Loyalists", "Bargain Seekers", "At-Risk")
  * Identify actionable insights for each segment
  * Recommend marketing/product strategies per segment

### 5. Validation & Robustness Checks
- [ ] Assess cluster stability:
  * Run clustering multiple times with different random seeds
  * Measure consistency of cluster assignments (e.g., adjusted rand index)
- [ ] Sensitivity to preprocessing:
  * Test with different scaling methods (MinMax vs Standard)
  * Test with different missing value imputation strategies
- [ ] Evaluate cluster separation:
  * Visualize using PCA/t-SNE for 2D projection if >2 features
  * Calculate pairwise distances between cluster centroids

### 6. Reporting & Deliverables
- [ ] Generate final report (`OLIST_CLUSTERING_REPORT.md`) containing:
  * Executive summary
  * Methodology overview
  * Optimal k selection justification (plots and metrics)
  * Detailed cluster profiles
  * Visualizations embedded in markdown
  * Business recommendations
- [ ] Save clustered dataset:
  * `olist_customer_features_with_clusters.csv` (customer features + cluster label)
- [ ] Save visualization assets:
  * PNG/JPG plots for inclusion in report
- [ ] Update README.md with clustering results and next steps

### 7. Cleanup & Reproducibility
- [ ] Ensure all scripts are in `scripts/` directory
- [ ] Document exact package versions used (requirements.txt or environment.yml)
- [ ] Commit code and results to Git repository with descriptive commit messages

## Expected Timeline
- Environment setup: 10 minutes
- Feature engineering: 15-30 minutes (depending on data size)
- Model selection & training: 5-15 minutes
- Analysis & reporting: 20-30 minutes
- Total: ~1 hour

## Risks & Mitigation
- **Risk**: Poor cluster separation (low silhouette scores)
  * Mitigation: Engineer additional features, try different clustering algorithms (DBSCAN, hierarchical)
- **Risk**: Dominant feature overwhelming clustering (e.g., monetary value)
  * Mitigation: Proper scaling, consider log-transform for monetary, evaluate feature importance
- **Risk**: Interpretation difficulty
  * Mitigation: Focus on actionable insights, validate with business stakeholders

## Success Criteria
- Clear, interpretable clusters with distinct business meanings
- Silhouette score > 0.25 (indicating reasonable separation)
- Actionable recommendations for marketing/product teams
- Reproducible pipeline documented in scripts

---
*Plan created for upcoming clustering session*