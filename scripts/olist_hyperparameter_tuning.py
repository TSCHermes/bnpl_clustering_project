import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import scipy.stats as stats
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
    
    # Preprocessing for numerical columns: RobustScaler
    # Preprocessing for categorical columns: OneHotEncoder (drop=None to keep all categories)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
    
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
    
    # Preprocess
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed feature shape: {X_train_processed.shape}")
    
    # Define models and parameter distributions
    models_params = [
        ('Gradient Boosting', 
         GradientBoostingClassifier(random_state=42),
         {
             'learning_rate': stats.uniform(0.01, 0.2),
             'n_estimators': stats.randint(100, 1000),
             'max_depth': stats.randint(3, 10),
             'subsample': stats.uniform(0.6, 0.4),
             'max_features': stats.uniform(0.6, 0.4)
         }),
        ('Random Forest', 
         RandomForestClassifier(random_state=42, n_jobs=-1),
         {
             'n_estimators': stats.randint(100, 1000),
             'max_depth': stats.randint(5, 30),
             'min_samples_split': stats.uniform(0.01, 0.5),
             'min_samples_leaf': stats.uniform(0.01, 0.5),
             'max_features': stats.uniform(0.6, 0.4)
         }),
        ('Logistic Regression', 
         LogisticRegression(random_state=42, max_iter=1000),
         {
             'C': stats.loguniform(0.01, 10.0),
             'penalty': ['l2'],
             'solver': ['lbfgs', 'liblinear']
         }),
        ('MLPClassifier', 
         MLPClassifier(random_state=42, max_iter=500),
         {
             'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
             'activation': ['relu', 'tanh'],
             'alpha': stats.loguniform(1e-5, 1e-1),
             'learning_rate_init': stats.loguniform(1e-4, 1e-1)
         })
    ]
    
    results = []
    
    for model_name, model, param_dist in models_params:
        print(f"\n=== Hyperparameter tuning for {model_name} ===")
        start_time = time.time()
        
        # Randomized search
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,
            scoring='roc_auc',
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train_processed, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate on test set
        y_pred_proba = search.best_estimator_.predict_proba(X_test_processed)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Also evaluate default model (with default parameters) for comparison
        default_model = model
        default_model.fit(X_train_processed, y_train)
        y_pred_proba_default = default_model.predict_proba(X_test_processed)[:, 1]
        default_auc = roc_auc_score(y_test, y_pred_proba_default)
        
        result = {
            'model': model_name,
            'best_params': search.best_params_,
            'best_cv_auc': search.best_score_,
            'test_auc': test_auc,
            'default_auc': default_auc,
            'train_time': train_time
        }
        results.append(result)
        
        print(f"Best CV AUC: {search.best_score_:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Default AUC: {default_auc:.4f}")
        print(f"Best params: {search.best_params_}")
        print(f"Time: {train_time:.2f}s")
    
    # Print summary
    print("\n=== Hyperparameter Tuning Summary ===")
    for res in results:
        print(f"{res['model']}:")
        print(f"  Default AUC: {res['default_auc']:.4f}")
        print(f"  Best CV AUC: {res['best_cv_auc']:.4f}")
        print(f"  Test AUC: {res['test_auc']:.4f}")
        print(f"  Improvement: {res['test_auc'] - res['default_auc']:.4f}")
        print()
    
    # Save results to file
    results_df = pd.DataFrame(results)
    results_path = data_path / 'docs' / 'OLIST_HYPERPARAMETER_TUNING_RESULTS.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()