"""
LEAD Dataset Explorer
Run this FIRST to understand your dataset structure
"""

import pandas as pd
import numpy as np
import os

def explore_file(filepath, name):
    """Explore a single CSV file"""
    print(f"\n{'='*70}")
    print(f"📄 {name}")
    print('='*70)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    # Get file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")
    
    # Load first few rows
    print("\n1. Loading sample (first 1000 rows)...")
    df = pd.read_csv(filepath, nrows=1000)
    
    # Basic info
    print(f"\n2. Dataset Info:")
    print(f"   Rows (sample): {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # Column names and types
    print(f"\n3. Columns ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].count()
        null_pct = (1 - non_null/len(df)) * 100
        print(f"   {i:2d}. {col:30s} | {str(dtype):10s} | Nulls: {null_pct:5.1f}%")
    
    # Sample data
    print(f"\n4. First 3 Rows:")
    print(df.head(3).to_string())
    
    # Statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n5. Numeric Column Statistics:")
        print(df[numeric_cols].describe().to_string())
    
    # Unique values for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0 and len(categorical_cols) < 10:
        print(f"\n6. Categorical Column Unique Values:")
        for col in categorical_cols[:5]:  # Show first 5
            unique_count = df[col].nunique()
            print(f"   {col}: {unique_count} unique values")
            if unique_count < 20:
                print(f"      Values: {df[col].unique()[:10].tolist()}")
    
    return df

def check_merge_compatibility(train_df, features_df):
    """Check how to merge train and features"""
    print(f"\n{'='*70}")
    print("🔗 MERGE ANALYSIS")
    print('='*70)
    
    train_cols = set(train_df.columns)
    features_cols = set(features_df.columns)
    
    common_cols = train_cols & features_cols
    train_only = train_cols - features_cols
    features_only = features_cols - train_cols
    
    print(f"\nCommon columns ({len(common_cols)}):")
    for col in common_cols:
        print(f"   - {col}")
    
    print(f"\nTrain-only columns ({len(train_only)}):")
    for col in sorted(train_only):
        print(f"   - {col}")
    
    print(f"\nFeatures-only columns ({len(features_only)}):")
    for col in sorted(features_only)[:20]:  # Show first 20
        print(f"   - {col}")
    if len(features_only) > 20:
        print(f"   ... and {len(features_only) - 20} more")
    
    # Suggest merge strategy
    print(f"\n💡 Merge Strategy:")
    if 'row_id' in common_cols:
        print("   ✓ Use 'row_id' as merge key")
        print("   Command: pd.merge(train, features, on='row_id')")
    elif 'id' in common_cols:
        print("   ✓ Use 'id' as merge key")
        print("   Command: pd.merge(train, features, on='id')")
    else:
        print("   ⚠️  No obvious merge key found")
        print("   Try: pd.concat([train, features], axis=1)")
        print("   Note: This assumes row alignment!")

def analyze_anomaly_distribution(df):
    """Analyze anomaly labels if present"""
    print(f"\n{'='*70}")
    print("🎯 ANOMALY ANALYSIS")
    print('='*70)
    
    anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() or 'label' in col.lower()]
    
    if not anomaly_cols:
        print("\n❌ No anomaly/label column found")
        print("   Looking for columns containing 'anomaly' or 'label'")
        return
    
    for col in anomaly_cols:
        print(f"\nColumn: {col}")
        print(f"Distribution:")
        value_counts = df[col].value_counts()
        for value, count in value_counts.items():
            percentage = count / len(df) * 100
            print(f"   {value}: {count:,} ({percentage:.2f}%)")

def main():
    print("\n" + "="*70)
    print(" LEAD DATASET EXPLORER")
    print("="*70)
    print("\nThis script will help you understand your dataset structure")
    print("before setting up the project.\n")
    
    # File paths
    files = {
        'train.csv': 'data/train.csv',
        'test.csv': 'data/test.csv',
        'train_features.csv': 'data/train_features.csv',
        'test_features.csv': 'data/test_features.csv',
        'sample_submission.csv': 'data/sample_submission.csv'
    }
    
    # Check which files exist
    print("📂 Checking for dataset files...")
    found_files = {}
    for name, path in files.items():
        if os.path.exists(path):
            print(f"   ✓ Found: {name}")
            found_files[name] = path
        else:
            print(f"   ✗ Missing: {name}")
    
    if not found_files:
        print("\n❌ No dataset files found in data/ directory")
        print("\n💡 Please:")
        print("   1. Download dataset from Kaggle")
        print("   2. Place files in data/ directory")
        print("   3. Run this script again")
        return
    
    # Explore each file
    dfs = {}
    for name, path in found_files.items():
        df = explore_file(path, name)
        if df is not None:
            dfs[name] = df
    
    # Special analysis
    if 'train.csv' in dfs and 'train_features.csv' in dfs:
        check_merge_compatibility(dfs['train.csv'], dfs['train_features.csv'])
    
    if 'train.csv' in dfs:
        analyze_anomaly_distribution(dfs['train.csv'])
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print('='*70)
    print(f"\nTotal files analyzed: {len(dfs)}")
    
    if 'train.csv' in dfs:
        print(f"\nDataset is ready for:")
        print("   ✓ MongoDB loading")
        print("   ✓ Feature engineering")
        print("   ✓ Model training")
        print("   ✓ Stream processing")
        print("\nNext step:")
        print("   python scripts/mongodb_setup.py")
    else:
        print("\n⚠️  train.csv not found. Please download from Kaggle.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()