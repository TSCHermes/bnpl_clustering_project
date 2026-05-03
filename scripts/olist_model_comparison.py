import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time

# Set data path
data_path = Path('/hermes_workspace/Olist_e_commerce_project')

def load_and_merge_data(sample_frac=0.05, random_state=42):
    """
    Load and merge the Olist datasets, return customer-level features and target.
    """
    print("Loading data...")
    # Load customers
    customers = pd.read_csv(data_path / 'data/raw' / 'olist_customers_dataset.csv', 
                            usecols=['customer_id', 'customer_unique_id', 'customer_state', 'customer_city'])
    # Sample customers
    sampled_customers = customers.sample(frac=sample_frac, random_state=random_state)
    print(f"Sampled {len(sampled_customers)} customers out of {len(customers)} ({sample_frac*100}%)")
    sampled_customer_ids = set(sampled_customers['customer_unique_id'])
    
    # Load orders
    orders = pd.read_csv(data_path / 'data/raw' / 'olist_orders_dataset.csv', 
                         usecols=['order_id', 'customer_id', 'order_purchase_timestamp'])
    orders_with_customer = pd.merge(orders, customers[['customer_id', 'customer_unique_id']], on='customer_id')
    orders_sampled = orders_with_customer[orders_with_customer['customer_unique_id'].isin(sampled_customer_ids)]
    print(f"Sampled {len(orders_sampled)} orders out of {len(orders)}")
    del orders, orders_with_customer
    
    # Load order items
    order_items = pd.read_csv(data_path / 'data/raw' / 'olist_order_items_dataset.csv', 
                              usecols=['order_id', 'order_item_id', 'product_id', 'seller_id', 'price', 'freight_value'])
    order_items_sampled = order_items[order_items['order_id'].isin(orders_sampled['order_id'])]
    print(f"Sampled {len(order_items_sampled)} order items out of {len(order_items)}")
    del order_items
    
    # Load order payments
    order_payments = pd.read_csv(data_path / 'data/raw' / 'olist_order_payments_dataset.csv', 
                                 usecols=['order_id', 'payment_value', 'payment_installments'])
    order_payments_sampled = order_payments[order_payments['order_id'].isin(orders_sampled['order_id'])]
    print(f"Sampled {len(order_payments_sampled)} order payments out of {len(order_payments)}")
    del order_payments
    
    # Load order reviews
    order_reviews = pd.read_csv(data_path / 'data/raw' / 'olist_order_reviews_dataset.csv', 
                                usecols=['order_id', 'review_score', 'review_comment_title', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp'])
    order_reviews_sampled = order_reviews[order_reviews['order_id'].isin(orders_sampled['order_id'])]
    print(f"Sampled {len(order_reviews_sampled)} order reviews out of {len(order_reviews)}")
    del order_reviews
    
    # Convert date columns
    orders_sampled['order_purchase_timestamp'] = pd.to_datetime(orders_sampled['order_purchase_timestamp'])
    order_reviews_sampled['review_creation_date'] = pd.to_datetime(order_reviews_sampled['review_creation_date'])
    order_reviews_sampled['review_answer_timestamp'] = pd.to_datetime(order_reviews_sampled['review_answer_timestamp'])
    
    # Merge data
    print("Merging data...")
    orders_customers = pd.merge(orders_sampled, 
                                customers[['customer_unique_id', 'customer_state', 'customer_city']], 
                                on='customer_unique_id')
    orders_items = pd.merge(orders_customers, 
                            order_items_sampled[['order_id', 'order_item_id', 'product_id', 'seller_id', 'price', 'freight_value']], 
                            on='order_id')
    
    # Products and category
    products = pd.read_csv(data_path / 'data/raw' / 'olist_products_dataset.csv', 
                           usecols=['product_id', 'product_category_name'])
    product_category_name_translation = pd.read_csv(data_path / 'data/raw' / 'product_category_name_translation.csv', 
                                                    usecols=['product_category_name', 'product_category_name_english'])
    products_with_category = pd.merge(products, product_category_name_translation[['product_category_name', 'product_category_name_english']], on='product_category_name', how='left')
    orders_items_products = pd.merge(orders_items, products_with_category[['product_id', 'product_category_name_english']], on='product_id')
    
    # Sellers
    sellers = pd.read_csv(data_path / 'data/raw' / 'olist_sellers_dataset.csv', 
                          usecols=['seller_id', 'seller_state'])
    orders_items_products_sellers = pd.merge(orders_items_products, sellers[['seller_id', 'seller_state']], on='seller_id', how='left')
    
    # Add payments
    orders_items_products_sellers_payments = pd.merge(orders_items_products_sellers, order_payments_sampled, on='order_id')
    
    # Add reviews
    orders_items_products_sellers_payments_reviews = pd.merge(orders_items_products_sellers_payments, order_reviews_sampled, on='order_id', how='left')
    
    print(f"Final merged data shape: {orders_items_products_sellers_payments_reviews.shape}")
    
    # Aggregate to customer level
    print("Aggregating to customer level...")
    latest_order_date = orders_items_products_sellers_payments_reviews['order_purchase_timestamp'].max()
    
    # Target: 1 if the customer has any order with payment_installments >= 2
    target_agg = orders_items_products_sellers_payments_reviews.groupby('customer_unique_id')['payment_installments'].max()
    target = (target_agg >= 2).astype(int)  # BNPL flag
    
    # Features: we will not include payment_installments in the features to avoid leakage
    features_agg = orders_items_products_sellers_payments_reviews.groupby('customer_unique_id').agg(
        # RFM features
        recency=('order_purchase_timestamp', lambda x: (latest_order_date - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('payment_value', 'sum'),
        # Additional behavioral features (excluding installments to avoid leakage in target)
        avg_monetary=('payment_value', 'mean'),
        avg_items_per_order=('order_item_id', 'mean'),
        avg_freight_per_order=('freight_value', 'mean'),
        avg_review_score=('review_score', 'mean'),
        pct_orders_with_review=('review_score', lambda x: x.notna().mean()),
        # Categorical features
        customer_state=('customer_state', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
        customer_city=('customer_city', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'),
        # Product diversity
        unique_product_categories=('product_category_name_english', 'nunique'),
        unique_sellers=('seller_id', 'nunique'),
    ).reset_index()
    
    # Merge target
    features_agg = features_agg.merge(target.rename('bnpl_target'), left_on='customer_unique_id', right_index=True)
    
    print(f"Final features shape: {features_agg.shape}")
    print(f"Target distribution: {features_agg['bnpl_target'].value_counts().to_dict()}")
    
    return features_agg

def prepare_features(df):
    """
    Prepare features for modeling: RFM + behavioral + geographic (Feature Set C)
    """
    # Separate features and target
    X = df.drop(['customer_unique_id', 'bnpl_target'], axis=1)
    y = df['bnpl_target']
    
    # Define column groups
    rfm_cols = ['recency', 'frequency', 'monetary']
    behavioral_cols = ['avg_monetary', 'avg_items_per_order', 'avg_freight_per_order', 
                       'avg_review_score', 'pct_orders_with_review', 
                       'unique_product_categories', 'unique_sellers']
    geographic_cols = ['customer_state', 'customer_city']
    
    selected_cols = rfm_cols + behavioral_cols + geographic_cols
    X_selected = X[selected_cols].copy()
    
    # Identify numerical and categorical columns
    num_cols = X_selected.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_selected.select_dtypes(include=['object']).columns.tolist()
    
    # Preprocessing for numerical columns: impute + RobustScaler
    # Preprocessing for categorical columns: impute + OneHotEncoder
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)])
    
    return X_selected, y, preprocessor

def main():
    # We'll use 5% sample for this classification
    sample_frac = 0.05
    print(f"Loading data with {sample_frac*100}% sample...")
    df = load_and_merge_data(sample_frac=sample_frac, random_state=42)
    
    # Prepare features (Feature Set C)
    X, y, preprocessor = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models with pipelines
    models = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ]),
        'HistGradientBoosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', HistGradientBoostingClassifier(random_state=42))
        ]),
        'MLPClassifier': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(random_state=42, max_iter=500))
        ])
    }
    
    results = []
    
    for name, pipeline in models.items():
        print(f"\n=== Training {name} ===")
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict probabilities
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        result = {
            'model': name,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'train_time': train_time
        }
        results.append(result)
        
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR:  {auc_pr:.4f}")
        print(f"Time: {train_time:.2f}s")
    
    # Print summary
    print("\n=== Model Comparison Summary ===")
    for res in results:
        print(f"{res['model']}: AUC-ROC={res['auc_roc']:.4f}, AUC-PR={res['auc_pr']:.4f}, Time={res['train_time']:.2f}s")
    
    # Save results to file
    results_df = pd.DataFrame(results)
    results_path = data_path / 'docs' / 'OLIST_MODEL_COMPARISON_RESULTS.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, pipeline in models.items():
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_roc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Olist BNPL Classification')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    roc_path = data_path / 'results' / 'visuals' / 'olist_model_comparison_roc.png'
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {roc_path}")
    
    # Feature importance for tree-based models
    for name in ['Random Forest', 'Gradient Boosting', 'HistGradientBoosting']:
        if name in models:
            pipeline = models[name]
            # Get feature names after preprocessing
            # We need to get the column names from the preprocessor
            # For simplicity, we can compute feature importances on the transformed data
            # but we don't have feature names easily. We'll skip for now or use a simpler approach.
            pass
    
    print("\nClassification analysis complete.")

if __name__ == "__main__":
    main()