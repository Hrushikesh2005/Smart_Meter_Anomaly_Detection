"""
Updated MongoDB Setup Script for ACTUAL LEAD Dataset Files
Works with: train.csv, test.csv, train_features.csv, test_features.csv
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
import pandas as pd
import numpy as np

class MongoDBSetup:
    def __init__(self, connection_string="mongodb://localhost:27017/"):
        self.client = MongoClient(connection_string)
        self.db = self.client['smart_meter_db']
        
    def create_collections(self):
        """Create all necessary collections with indexes"""
        
        # Collection 1: Meter Readings (from train.csv)
        meter_readings = self.db['meter_readings']
        meter_readings.create_index([
            ('building_id', ASCENDING),
            ('timestamp', ASCENDING)
        ])
        meter_readings.create_index([('timestamp', DESCENDING)])
        meter_readings.create_index([('anomaly', ASCENDING)])
        print("✓ Created 'meter_readings' collection with indexes")
        
        # Collection 2: Features (from train_features.csv)
        features = self.db['features']
        features.create_index([
            ('building_id', ASCENDING),
            ('timestamp', ASCENDING)
        ])
        print("✓ Created 'features' collection with indexes")
        
        # Collection 3: Detected Anomalies
        anomalies = self.db['anomalies']
        anomalies.create_index([
            ('building_id', ASCENDING),
            ('timestamp', DESCENDING)
        ])
        anomalies.create_index([('anomaly_score', DESCENDING)])
        print("✓ Created 'anomalies' collection with indexes")
        
        # Collection 4: Model Performance
        model_performance = self.db['model_performance']
        model_performance.create_index([('timestamp', DESCENDING)])
        print("✓ Created 'model_performance' collection with indexes")
        
        # Collection 5: Stream Processing Stats
        stream_stats = self.db['stream_stats']
        stream_stats.create_index([('timestamp', DESCENDING)])
        print("✓ Created 'stream_stats' collection with indexes")
        
        print("\n✅ All collections created successfully!")
        
    def load_training_data(self, train_path, train_features_path):
        """Load ACTUAL LEAD dataset into MongoDB"""
        
        print("\n📥 Loading LEAD dataset into MongoDB...")
        print("="*60)
        
        # Load train.csv (meter readings with anomaly labels)
        print("\n1. Loading train.csv...")
        train_df = pd.read_csv(train_path)
        print(f"   Loaded {len(train_df):,} records")
        print(f"   Columns: {list(train_df.columns)}")
        
        # Load train_features.csv
        print("\n2. Loading train_features.csv...")
        features_df = pd.read_csv(train_features_path)
        print(f"   Loaded {len(features_df):,} records")
        print(f"   Columns: {list(features_df.columns)}")
        
        # Merge train and features
        print("\n3. Merging datasets...")
        # Identify common columns (likely 'row_id' or index-based)
        if 'row_id' in train_df.columns and 'row_id' in features_df.columns:
            merged_df = train_df.merge(features_df, on='row_id', how='left')
        elif 'id' in train_df.columns and 'id' in features_df.columns:
            merged_df = train_df.merge(features_df, on='id', how='left')
        else:
            # Merge by index
            merged_df = pd.concat([train_df, features_df], axis=1)
        
        print(f"   Merged dataset: {len(merged_df):,} records with {len(merged_df.columns)} features")
        
        # Convert timestamp if present
        timestamp_cols = [col for col in merged_df.columns if 'time' in col.lower()]
        if timestamp_cols:
            print(f"\n4. Converting timestamp column: {timestamp_cols[0]}")
            merged_df[timestamp_cols[0]] = pd.to_datetime(merged_df[timestamp_cols[0]])
        
        # Insert into MongoDB in batches
        print("\n5. Inserting into MongoDB...")
        batch_size = 10000
        total_inserted = 0
        
        for i in range(0, len(merged_df), batch_size):
            batch = merged_df.iloc[i:i+batch_size]
            records = batch.to_dict('records')
            
            # Replace NaN with None for MongoDB
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
            
            self.db['meter_readings'].insert_many(records)
            total_inserted += len(records)
            
            if (i + batch_size) % 50000 == 0:
                print(f"   Inserted {total_inserted:,} / {len(merged_df):,} records ({total_inserted/len(merged_df)*100:.1f}%)")
        
        print(f"\n✓ Successfully loaded {total_inserted:,} meter readings")
        
        # Show sample record
        print("\n📋 Sample Record:")
        sample = merged_df.iloc[0].to_dict()
        for key, value in list(sample.items())[:10]:  # Show first 10 fields
            print(f"   {key}: {value}")
        if len(sample) > 10:
            print(f"   ... and {len(sample) - 10} more fields")
        
        print("\n✅ Data loading complete!")
        
    def get_collection_stats(self):
        """Display statistics about collections"""
        
        print("\n📊 Database Statistics:")
        print("="*60)
        
        collections = [
            'meter_readings',
            'features',
            'anomalies',
            'model_performance',
            'stream_stats'
        ]
        
        for coll_name in collections:
            count = self.db[coll_name].count_documents({})
            print(f"{coll_name:.<40} {count:>10,} documents")
        
        # Anomaly statistics
        if self.db['meter_readings'].count_documents({}) > 0:
            print("\n📊 Anomaly Distribution:")
            pipeline = [
                {
                    '$group': {
                        '_id': '$anomaly',
                        'count': {'$sum': 1}
                    }
                }
            ]
            anomaly_stats = list(self.db['meter_readings'].aggregate(pipeline))
            for stat in anomaly_stats:
                label = "Normal (0)" if stat['_id'] == 0 else "Anomaly (1)"
                print(f"   {label}: {stat['count']:,}")
    
    def analyze_dataset_structure(self, train_path, features_path):
        """Analyze the actual structure of LEAD dataset files"""
        
        print("\n🔍 Analyzing LEAD Dataset Structure...")
        print("="*60)
        
        # Analyze train.csv
        print("\n📄 train.csv:")
        train_df = pd.read_csv(train_path, nrows=5)
        print(f"   Shape: {train_df.shape}")
        print(f"   Columns: {list(train_df.columns)}")
        print(f"   Data types:\n{train_df.dtypes}")
        print(f"\n   Sample rows:")
        print(train_df.head(3))
        
        # Analyze train_features.csv
        print("\n\n📄 train_features.csv:")
        features_df = pd.read_csv(features_path, nrows=5)
        print(f"   Shape: {features_df.shape}")
        print(f"   Columns: {list(features_df.columns)}")
        print(f"   Data types:\n{features_df.dtypes}")
        print(f"\n   Sample rows:")
        print(features_df.head(3))
        
        # Find common columns
        train_cols = set(train_df.columns)
        feature_cols = set(features_df.columns)
        common = train_cols & feature_cols
        
        print(f"\n\n🔗 Common columns: {list(common)}")
        print(f"   Train-only columns: {list(train_cols - feature_cols)}")
        print(f"   Features-only columns: {list(feature_cols - train_cols)}")

if __name__ == "__main__":
    print("🚀 Smart Meter Anomaly Detection - MongoDB Setup (LEAD Dataset)")
    print("="*60)
    
    # Initialize
    mongo = MongoDBSetup()
    
    # First, analyze dataset structure
    print("\n⚠️  IMPORTANT: First analyzing your dataset structure...")
    mongo.analyze_dataset_structure(
        'data/train.csv',
        'data/train_features.csv'
    )
    
    print("\n" + "="*60)
    input("\n Press Enter to continue with database setup...")
    
    # Create collections and indexes
    mongo.create_collections()
    
    # Load data
    print("\n⚠️  This will load data into MongoDB (may take 5-10 minutes)")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        mongo.load_training_data(
            'data/train.csv',
            'data/train_features.csv'
        )
        
        # Show stats
        mongo.get_collection_stats()
    
    print("\n✅ Setup complete!")