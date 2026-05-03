import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set data path
data_path = Path('/hermes_workspace/Olist_e_commerce_project')

def load_and_merge_data(sample_frac=0.01, random_state=42):
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
    
    # We'll compute the target: BNPL flag based on max installments >=2
    # First, compute agg for features and target separately to avoid leakage
    # For features, we don't want to use the installments feature that we use for target? 
    # We'll compute target first.
    
    # Target: 1 if the customer has any order with payment_installments >= 2
    target_agg = orders_items_products_sellers_payments_reviews.groupby('customer_unique_id')['payment_installments'].max()
    target = (target_agg >= 2).astype(int)  # BNPL flag
    
    # Features: we will not include payment_installments in the features to avoid leakage
    # We'll compute other aggregates
    features_agg = orders_items_products_sellers_payments_reviews.groupby('customer_unique_id').agg(
        # RFM features
        recency=('order_purchase_timestamp', lambda x: (latest_order_date - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('payment_value', 'sum'),
        # Additional behavioral features (excluding avg_installments to avoid leakage in target)
        avg_monetary=('payment_value', 'mean'),
        avg_items_per_order=('order_item_id', 'mean'),
        avg_freight_per_order=('freight_value', 'mean'),
        # avg_installments is excluded because we used it for target
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

def prepare_features(df, feature_set='C'):
    """
    Prepare features for modeling based on the feature set.
    feature_set: 
        'A': RFM only (recency, frequency, monetary)
        'B': RFM + behavioral (excluding avg_installments to avoid leakage)
        'C': Set B + geographic (one-hot encoded top 10 states/cities)
        'D': Set C + PCA (apply PCA on numerical features of Set C)
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
    
    if feature_set == 'A':
        selected_cols = rfm_cols
    elif feature_set == 'B':
        selected_cols = rfm_cols + behavioral_cols
    elif feature_set == 'C':
        selected_cols = rfm_cols + behavioral_cols + geographic_cols
    elif feature_set == 'D':
        selected_cols = rfm_cols + behavioral_cols + geographic_cols
    else:
        raise ValueError("Invalid feature set. Choose from 'A', 'B', 'C', 'D'.")
    
    X_selected = X[selected_cols].copy()
    
    # Identify numerical and categorical columns
    num_cols = X_selected.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_selected.select_dtypes(include=['object']).columns.tolist()
    
    # For geographic columns, we will frequency encode or one-hot encode top categories to avoid high dimensionality
    # We'll do: for each geographic column, keep top 10 categories, rest as 'Other'
    if 'customer_state' in cat_cols:
        state_counts = X_selected['customer_state'].value_counts()
        top10_states = state_counts.head(10).index.tolist()
        X_selected['customer_state'] = X_selected['customer_state'].apply(lambda x: x if x in top10_states else 'Other')
        # Update cat_cols: it's still categorical but now with fewer categories
    if 'customer_city' in cat_cols:
        city_counts = X_selected['customer_city'].value_counts()
        top10_cities = city_counts.head(10).index.tolist()
        X_selected['customer_city'] = X_selected['customer_city'].apply(lambda x: x if x in top10_cities else 'Other')
    
    # Update cat_cols after potential changes
    cat_cols = X_selected.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_selected.select_dtypes(include=[np.number]).columns.tolist()
    
    # Preprocessing for numerical columns: standardize
    # Preprocessing for categorical columns: one-hot encode
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
    
    if feature_set == 'D':
        # We will apply PCA after preprocessing, but we need to do it in a pipeline
        # We'll create a pipeline that does preprocessing then PCA
        # However, we want to compare with and without PCA, so we'll handle it separately in the modeling loop
        # For now, return the preprocessor and the data, and we'll add PCA in the model pipeline if needed
        pass
    
    return X_selected, y, preprocessor, num_cols, cat_cols

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_set):
    """
    Train and evaluate a model, return metrics.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        'model': model_name,
        'feature_set': feature_set,
        'auc': auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'train_time': train_time,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }

def main():
    # We'll use 1% sample for this classification to speed up
    sample_frac = 0.01
    print(f"Loading data with {sample_frac*100}% sample...")
    df = load_and_merge_data(sample_frac=sample_frac, random_state=42)
    
    # Define feature sets and models
    feature_sets = ['A', 'B', 'C', 'D']
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=50, eval_metric='logloss', n_jobs=-1)
    }
    
    results = []
    # We'll store the best model's predictions for ROC curve plotting
    best_auc = 0
    best_model_info = None
    
    for fs in feature_sets:
        print(f"\n=== Processing Feature Set {fs} ===")
        X, y, preprocessor, num_cols, cat_cols = prepare_features(df, feature_set=fs)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Preprocess
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # For feature set D, we will apply PCA after preprocessing
        if fs == 'D':
            # Determine number of components to retain 95% variance
            pca = PCA(n_components=0.95, random_state=42)
            X_train_processed = pca.fit_transform(X_train_processed)
            X_test_processed = pca.transform(X_test_processed)
            print(f"  After PCA: {X_train_processed.shape[1]} features")
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            res = evaluate_model(model, X_train_processed, X_test_processed, y_train, y_test, model_name, fs)
            results.append(res)
            
            # Track best model for ROC curve
            if res['auc'] > best_auc:
                best_auc = res['auc']
                best_model_info = {
                    'model_name': model_name,
                    'feature_set': fs,
                    'model': model,
                    'X_test_processed': X_test_processed,
                    'y_test': y_test,
                    'y_pred_proba': res['y_pred_proba']
                }
            
            # Print immediate feedback
            print(f"    AUC: {res['auc']:.4f}, Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f}")
    
    # Convert results to DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    print("\n=== Results Summary ===")
    print(results_df[['model', 'feature_set', 'auc', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))
    
    # Save results to file
    results_path = data_path / 'docs' / 'OLIST_CLASSIFICATION_RESULTS.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Plot ROC curve for the best model
    if best_model_info is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(best_model_info['y_test'], best_model_info['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{best_model_info['model_name']} (FS {best_model_info['feature_set']}) (AUC = {best_auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Best Model')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        roc_path = data_path / 'results/visuals' / 'olist_classification_roc_curve.png'
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"ROC curve saved to {roc_path}")
    
    print("\nClassification analysis complete.")

if __name__ == "__main__":
    main()