import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, KBinsDiscretizer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
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
        # Additional behavioral features (excluding installments to avoid leakage in target)
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

def get_preprocessing_configs():
    configs = []
    
    # Config 1: Basic - scale numeric, one-hot encode categorical
    preprocess1 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), []),  # placeholder, will be set later
            ('cat', OneHotEncoder(handle_unknown='ignore'), [])
        ])
    configs.append(("StandardScale_OneHot", preprocess1))
    
    # Config 2: Scale numeric with RobustScaler, one-hot encode categorical
    preprocess2 = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), []),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [])
        ])
    configs.append(("RobustScale_OneHot", preprocess2))
    
    # Config 3: Scale numeric, one-hot encode categorical, drop first category to avoid multicollinearity
    preprocess3 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), []),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), [])
        ])
    configs.append(("StandardScale_OneHotDropFirst", preprocess3))
    
    # Config 4: Bin numeric features into quantiles, one-hot encode categorical
    preprocess4 = ColumnTransformer(
        transformers=[
            ('num', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'), []),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [])
        ])
    configs.append(("KBinsDiscretizer_OneHot", preprocess4))
    
    # Config 5: Polynomial features for numeric (degree 2), one-hot encode categorical
    preprocess5 = ColumnTransformer(
        transformers=[
            ('num', PolynomialFeatures(degree=2, include_bias=False), []),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [])
        ])
    configs.append(("Poly2_OneHot", preprocess5))
    
    return configs

def get_models():
    models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
        ("KNeighbors_5", KNeighborsClassifier(n_neighbors=5))
    ]
    return models

def evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocess, numeric_features, categorical_features):
    """Train model and return metrics"""
    # Update preprocess with actual feature lists
    preprocess_transformed = ColumnTransformer(
        transformers=[
            ('num', preprocess.named_transformers_['num'], numeric_features) if hasattr(preprocess, 'named_transformers_') else (preprocess.named_transformers_['num'], numeric_features),
            ('cat', preprocess.named_transformers_['cat'], categorical_features) if hasattr(preprocess, 'named_transformers_') else (preprocess.named_transformers_['cat'], categorical_features)
        ]
    ) if False else None  # We'll rebuild the preprocessor below
    
    # Actually, we need to rebuild the preprocessor with the correct columns
    # The preprocess passed in is a template, we need to set the columns
    # Let's reconstruct:
    if hasattr(preprocess, 'named_transformers_'):
        # It's already a ColumnTransformer, we can't change columns easily, so we'll rebuild
        pass
    
    # Rebuild the preprocessor with the actual column names
    # We'll get the type of each step from the template
    # This is a bit messy, so let's change approach: we'll pass the preprocessing steps as classes and params
    # Instead, we'll do: for each config, we know what it is, so we can rebuild with columns
    # But for simplicity, we'll rebuild inside the loop for each config and model? 
    # Actually, we already have the configs as ColumnTransformer with empty column lists.
    # We'll update the column lists in the preprocess object.
    # However, ColumnTransformer doesn't allow changing columns after creation? 
    # We'll create a new ColumnTransformer for each config with the correct columns.
    
    # Let's change the approach: we'll pass the preprocessing steps as a list of (name, transformer, columns) 
    # But to keep changes minimal, we'll do:
    #   For each config name and preprocess template, we create a new ColumnTransformer with the same transformers but with the actual columns.
    
    # Extract the transformers from the template
    transformers = []
    for name, trans, cols in preprocess.transformers:
        # cols is empty, we will replace with the actual columns
        if name == 'num':
            transformers.append((name, trans, numeric_features))
        elif name == 'cat':
            transformers.append((name, trans, categorical_features))
    preprocess_actual = ColumnTransformer(transformers=transformers)
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocess_actual),
        ('classifier', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities for AUC-PR
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate AUC-PR
    ap_score = average_precision_score(y_test, y_pred_proba)
    
    # Also get predictions for other metrics
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model_name': name,
        'auc_pr': ap_score,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'pipeline': pipeline  # store for potential reuse
    }

def main():
    # We'll use 5% sample for this classification (to match the Olist classification script)
    sample_frac = 0.05
    print(f"Loading data with {sample_frac*100}% sample...")
    df = load_and_merge_data(sample_frac=sample_frac, random_state=42)
    
    # Create binary target: 0 = No BNPL, 1 = BNPL
    # Already done in load_and_merge_data
    X = df.drop(['customer_unique_id', 'bnpl_target'], axis=1)
    y = df['bnpl_target']
    
    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    if len(numeric_features) > 0:
        print(f"Numeric features sample: {numeric_features[:5]}")
    if len(categorical_features) > 0:
        print(f"Categorical features sample: {categorical_features[:5]}")
    
    # Define 5 different preprocessing configurations
    def get_preprocessing_configs():
        configs = []
        
        # Config 1: Basic - scale numeric, one-hot encode categorical
        preprocess1 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        configs.append(("StandardScale_OneHot", preprocess1))
        
        # Config 2: Scale numeric with RobustScaler, one-hot encode categorical
        preprocess2 = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        configs.append(("RobustScale_OneHot", preprocess2))
        
        # Config 3: Scale numeric, one-hot encode categorical, drop first category to avoid multicollinearity
        preprocess3 = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
            ])
        configs.append(("StandardScale_OneHotDropFirst", preprocess3))
        
        # Config 4: Bin numeric features into quantiles, one-hot encode categorical
        preprocess4 = ColumnTransformer(
            transformers=[
                ('num', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        configs.append(("KBinsDiscretizer_OneHot", preprocess4))
        
        # Config 5: Polynomial features for numeric (degree 2), one-hot encode categorical
        preprocess5 = ColumnTransformer(
            transformers=[
                ('num', PolynomialFeatures(degree=2, include_bias=False), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        configs.append(("Poly2_OneHot", preprocess5))
        
        return configs
    
    # Define 4 models (total experiments = 5*4 = 20)
    def get_models():
        models = [
            ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
            ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
            ("KNeighbors_5", KNeighborsClassifier(n_neighbors=5))
        ]
        return models
    
    configs = get_preprocessing_configs()
    models = get_models()
    
    print(f"Number of preprocessing configurations: {len(configs)}")
    print(f"Number of models: {len(models)}")
    print(f"Total experiments: {len(configs) * len(models)}")
    
    # Store all results
    all_results = []
    
    # Loop over configurations and models
    for config_name, preprocess in configs:
        for model_name, model in models:
            print(f"\nRunning: {config_name} + {model_name}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            try:
                result = evaluate_model(model_name, model, X_train, X_test, y_train, y_test, preprocess, numeric_features, categorical_features)
                all_results.append(result)
                
                # Generate individual report
                REPORTS_DIR = '/hermes_workspace/Olist_e_commerce_project/results/model_comparison'
                os.makedirs(REPORTS_DIR, exist_ok=True)
                report_path = os.path.join(REPORTS_DIR, f"report_{config_name}_{model_name}.md")
                with open(report_path, 'w') as f:
                    f.write(f"# Experiment Report: {config_name} + {model_name}\n\n")
                    f.write(f"**Configuration**: {config_name}\n")
                    f.write(f"**Model**: {model_name}\n\n")
                    f.write(f"## Metrics\n")
                    f.write(f"- AUC-PR: {result['auc_pr']:.4f}\n")
                    f.write(f"- Precision: {result['precision']:.4f}\n")
                    f.write(f"- Recall: {result['recall']:.4f}\n")
                    f.write(f"- F1-Score: {result['f1']:.4f}\n\n")
                    f.write(f"## Configuration Description\n")
                    f.write(f"This configuration used: {config_name}\n")
                    f.write(f"## Model Description\n")
                    f.write(f"This experiment used: {model_name} with default/hyperparameters as specified in the script.\n")
                
                print(f"  AUC-PR: {result['auc_pr']:.4f}")
                
            except Exception as e:
                print(f"  Error with {model_name}: {str(e)}")
                # Still create a report for failed runs
                REPORTS_DIR = '/hermes_workspace/Olist_e_commerce_project/results/model_comparison'
                os.makedirs(REPORTS_DIR, exist_ok=True)
                report_path = os.path.join(REPORTS_DIR, f"report_{config_name}_{model_name}_FAILED.md")
                with open(report_path, 'w') as f:
                    f.write(f"# Failed Experiment: {config_name} + {model_name}\n\n")
                    f.write(f"**Error**: {str(e)}\n")
                continue
    
    # Sort results by AUC-PR descending
    all_results.sort(key=lambda x: x['auc_pr'], reverse=True)
    
    # Generate summary report
    REPORTS_DIR = '/hermes_workspace/Olist_e_commerce_project/results/model_comparison'
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary_path = os.path.join(REPORTS_DIR, "SUMMARY.md")
    with open(summary_path, 'w') as f:
        f.write("# Olist E-commerce Model Experiment Summary (BNPL Style)\n\n")
        f.write(f"**Total Experiments**: {len(all_results)}\n")
        f.write(f"**Successful Experiments**: {len(all_results)}\n\n")
        f.write("## Top 10 Models by AUC-PR\n\n")
        f.write("| Rank | Configuration | Model | AUC-PR | Precision | Recall | F1 |\n")
        f.write("|------|---------------|-------|--------|-----------|--------|----|\n")
        for i, res in enumerate(all_results[:10]):
            f.write(f"| {i+1} | {res['config_name']} | {res['model_name']} | {res['auc_pr']:.4f} | {res['precision']:.4f} | {res['recall']:.4f} | {res['f1']:.4f} |\n")
        
        f.write("\n## Best Overall Model\n\n")
        if all_results:
            best = all_results[0]
            f.write(f"- **Configuration**: {best['config_name']}\n")
            f.write(f"- **Model**: {best['model_name']}\n")
            f.write(f"- **AUC-PR**: {best['auc_pr']:.4f}\n")
            f.write(f"- **Precision**: {best['precision']:.4f}\n")
            f.write(f"- **Recall**: {best['recall']:.4f}\n")
            f.write(f"- **F1-Score**: {best['f1']:.4f}\n\n")
            f.write("### Configuration Details\n")
            f.write(f"The best configuration was: {best['config_name']}\n")
            f.write("### Model Details\n")
            f.write(f"The best model was: {best['model_name']}\n")
        else:
            f.write("No successful experiments.\n")
        
        f.write("\n## All Reports\n")
        f.write(f"Individual reports are stored in the `{REPORTS_DIR}` directory.\n")
        f.write(f"Total reports generated: {len([f for f in os.listdir(REPORTS_DIR) if f.startswith('report_') and f.endswith('.md')])}\n")
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    print(f"Results saved to: {REPORTS_DIR}")
    print(f"Summary: {summary_path}")
    if all_results:
        print(f"Top model AUC-PR: {all_results[0]['auc_pr']:.4f} ({all_results[0]['config_name']} + {all_results[0]['model_name']})")
    print("="*50)

if __name__ == "__main__":
    main()