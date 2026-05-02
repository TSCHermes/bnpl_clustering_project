import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Define data path
data_path = Path('/hermes_workspace/Olist_e_commerce_project')

# List of CSV files
csv_files = [f for f in data_path.glob('*.csv') if f.name != 'Kaggle Page.txt']
print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {f.name}")

# Dictionary to hold dataframes and their info
data_info = {}

# Function to get basic info about a dataframe
def get_df_info(df, name):
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'sample': df.head(3).to_dict() if len(df) > 0 else {}
    }
    return info

# Load each CSV and store info
for csv_file in csv_files:
    print(f"\nLoading {csv_file.name}...")
    try:
        # For large files, we might want to read only a subset for initial inspection?
        # But let's try to read the whole thing; we can always sample later.
        df = pd.read_csv(csv_file)
        data_info[csv_file.name] = get_df_info(df, csv_file.name)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        # Show missing values summary
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            print(f"  Missing values in {len(missing_cols)} columns:")
            for col, count in missing_cols.items():
                print(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print(f"  No missing values")
    except Exception as e:
        print(f"  Error loading {csv_file.name}: {e}")
        data_info[csv_file.name] = {'error': str(e)}

# Now generate a report
report_lines = []
report_lines.append("# Olist E-commerce Dataset - Exploratory Data Analysis Report\n")
report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

report_lines.append("## Overview\n")
report_lines.append(f"The Olist E-commerce dataset consists of {len(csv_files)} CSV files relating to the Brazilian e-commerce platform Olist. Below is a summary of each file.\n\n")

for file_name, info in data_info.items():
    if 'error' in info:
        report_lines.append(f"### {file_name}\n")
        report_lines.append(f"**Error**: {info['error']}\n\n")
        continue
    report_lines.append(f"### {file_name}\n")
    report_lines.append(f"- **Shape**: {info['shape'][0]} rows, {info['shape'][1]} columns\n")
    report_lines.append(f"- **Columns**: {', '.join(info['columns'])}\n")
    report_lines.append("\n#### Missing Values\n")
    missing_pct = info['missing_percentage']
    missing_cols = {k: v for k, v in missing_pct.items() if v > 0}
    if missing_cols:
        report_lines.append("| Column | Missing Count | Missing Percentage |\n")
        report_lines.append("|--------|---------------|-------------------|\n")
        for col, pct in missing_cols.items():
            count = info['missing_values'][col]
            report_lines.append(f"| {col} | {count} | {pct:.2f}% |\n")
    else:
        report_lines.append("No missing values.\n")
    report_lines.append("\n#### Data Types\n")
    report_lines.append("| Column | Data Type |\n")
    report_lines.append("|--------|-----------|\n")
    for col, dtype in info['dtypes'].items():
        report_lines.append(f"| {col} | {dtype} |\n")
    report_lines.append("\n#### First 3 Rows (Sample)\n")
    sample = info['sample']
    if sample:
        # Convert sample dict to a DataFrame for nice display? We'll just list.
        for i in range(len(next(iter(sample.values())))):
            report_lines.append(f"**Row {i+1}**: ")
            row_vals = [f"{col}: {sample[col][i]}" for col in sample.keys()]
            report_lines.append("; ".join(row_vals) + "\n")
    else:
        report_lines.append("No data.\n")
    report_lines.append("\n---\n\n")

# Also consider relationships between tables? We'll note key identifiers.
report_lines.append("## Key Identifiers for Joining\n")
report_lines.append("- `customer_id`: appears in `olist_customers_dataset.csv` and `olist_orders_dataset.csv`\n")
report_lines.append("- `order_id`: appears in `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_payments_dataset.csv`, `olist_order_reviews_dataset.csv`\n")
report_lines.append("- `product_id`: appears in `olist_order_items_dataset.csv` and `olist_products_dataset.csv`\n")
report_lines.append("- `seller_id`: appears in `olist_order_items_dataset.csv` and `olist_sellers_dataset.csv`\n")
report_lines.append("- `review_id`: appears in `olist_order_reviews_dataset.csv`\n")
report_lines.append("- `product_category_name`: appears in `olist_products_dataset.csv` and `product_category_name_translation.csv`\n")
report_lines.append("- `geolocation_zip_code_prefix`: appears in `olist_geolocation_dataset.csv` and matches `customer_zip_code_prefix` and `seller_zip_code_prefix` (in customers and sellers datasets)\n\n")

report_lines.append("## Suggested Next Steps for Clustering\n")
report_lines.append("1. Create customer-level RFM (Recency, Frequency, Monetary) features from orders and payments.\n")
report_lines.append("2. Consider order-level features: ticket size, number of items, payment installments, review score.\n")
report_lines.append("3. Product-level features: category, price, weight, volume.\n")
report_lines.append("4. Seller-level features: number of customers, geographic distribution.\n")
report_lines.append("5. Geolocation features: state, city, zip code prefix.\n")
report_lines.append("6. Handle missing values appropriately (e.g., impute or drop).\n")
report_lines.append("7. Encode categorical variables (state, category, etc.) for clustering algorithms.\n")
report_lines.append("8. Scale numerical features before applying clustering algorithms like KMeans.\n")

# Write report to file
report_path = data_path / 'OLIST_EDA_REPORT.md'
with open(report_path, 'w') as f:
    f.writelines(report_lines)

print(f"\nEDA report written to {report_path}")