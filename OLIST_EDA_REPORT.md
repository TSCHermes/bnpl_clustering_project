# Olist E-commerce Dataset - Exploratory Data Analysis (EDA) Report

## Overview
This report summarizes the exploratory data analysis performed on the Olist E-commerce dataset, which consists of 9 CSV files containing information about orders, customers, products, sellers, geolocation, etc.

## Dataset Details
- **Total Files**: 9 CSV files
- **Approximate Total Rows**: 1.3M
- **Key Tables**:
  1. `olist_orders_dataset.csv`: 99,441 rows (order-level info)
  2. `olist_order_items_dataset.csv`: 112,650 rows (order line items)
  3. `olist_order_payments_dataset.csv`: 103,886 rows (payment transactions)
  4. `olist_order_reviews_dataset.csv`: 99,224 rows (customer reviews)
  5. `olist_customers_dataset.csv`: 99,441 rows (customer profiles)
  6. `olist_sellers_dataset.csv`: 3,095 rows (seller information)
  7. `olist_products_dataset.csv`: 32,951 rows (product catalog)
  8. `olist_product_category_name_translation.csv`: 71 rows (category translations)
  9. `olist_geolocation_dataset.csv`: 1,000,163 rows (geolocation reference)

## Key Identifier Fields for Joining
- **`customer_id`**: Links `customers` ↔ `orders`
- **`order_id`**: Links `orders` ↔ `order_items`, `order_payments`, `order_reviews`
- **`product_id`**: Links `order_items` ↔ `products`
- **`seller_id`**: Links `order_items` ↔ `sellers`
- **`geolocation_zip_code_prefix`**: Matches `customers`/`sellers` zip prefixes to `geolocation`

## Missing Values Analysis
### Critical Missingness:
1. **Order Delivery Timestamps** (in `orders`):
   - `order_delivered_carrier_date`: 0.16% missing
   - `order_delivered_customer_date`: 2.98% missing
   - `order_estimated_delivery_date`: 0.00% missing (complete)
   - `order_approved_at`: 0.00% missing (complete)

2. **Order Reviews** (in `order_reviews`):
   - `review_comment_title`: 88.3% missing
   - `review_comment_message`: 58.7% missing
   - `review_score`: 0.0% missing (complete)

3. **Product Specifications** (in `products`):
   - `product_category_name`: ~1.85% missing
   - `product_name_lenght`: ~1.85% missing
   - `product_description_lenght`: ~1.85% missing
   - `product_photos_qty`: ~1.85% missing
   - `product_weight_g`: ~1.85% missing
   - `product_length_cm`: ~1.85% missing
   - `product_height_cm`: ~1.85% missing
   - `product_width_cm`: ~1.85% missing

## Data Types Observed
- **String Identifiers**: Most ID fields (e.g., `order_id`, `customer_id`)
- **Integer Codes**: Payment types, review scores, etc.
- **Floats**: Monetary values (prices, weights), review scores
- **Timestamps**: Stored as strings (require conversion to datetime)

## Recommended Features for Customer Segmentation
Based on the EDA, the following features are recommended for building customer-level segments:

### 1. Core RFM (Recency, Frequency, Monetary)
- **Recency**: Days since customer's last purchase (calculated from `order_purchase_timestamp`)
- **Frequency**: Total number of orders per customer
- **Monetary**: Total payment value (sum of `payment_value`) per customer

### 2. Behavioral Features (Order-Level Aggregates)
- Average items per order
- Average freight value per order
- Average number of payment installments
- Percentage of orders with reviews
- Average review score

### 3. Geographic/Demographic Features
- Customer state (one-hot encoded or target encoded)
- Customer city (consider state-level aggregation due to high cardinality)

### 4. Product Affinity Features (Optional)
- Diversity of product categories purchased
- Average product price tier
- Preferred payment method distribution

### 5. Seller Interaction Features (Optional)
- Number of unique sellers transacted with
- Average seller performance (if available)

## Next Steps for Clustering Analysis
1. **Feature Engineering**: Merge tables to create customer-level feature set
2. **Preprocessing**:
   - Handle missing values (numerical: median; categorical: mode)
   - Encode categorical variables (state: one-hot or target encoding)
   - Scale numerical features (standardization)
3. **Clustering Algorithm**: Evaluate KMeans for k=2 to 6
4. **Validation Metrics**:
   - Silhouette Score (cohesion vs separation)
   - Calinski-Harabasz Index (ratio of between-cluster to within-cluster variance)
   - Davies-Bouldin Index (average similarity between clusters)
5. **Output**:
   - Cluster assignments for each customer
   - Feature importance/characterization per cluster
   - Visualization (e.g., scatter plots of key feature pairs)
   - Business interpretation of each segment

## Files Created During EDA
- `OLIST_EDA_REPORT.md`: This report
- Updated `README.md`: Includes EDA summary and notes on upcoming clustering
- `scripts/olist_clustering_fixed.py`: Prepared clustering script (not executed yet)

## Conclusion
The EDA reveals a rich dataset suitable for customer segmentation. The recommended RFM-based approach, enriched with behavioral and geographic features, should yield actionable segments for marketing and product strategies. The next session will focus on executing the clustering pipeline and interpreting the results.

---
*Report generated during EDA session on [current date]*