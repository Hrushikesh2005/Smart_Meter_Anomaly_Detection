"""
Fixed MongoDB Setup Script for LEAD Dataset
Handles duplicate columns properly
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
import pandas as pd
import numpy as np
import os

class MongoDBSetup:
    def __init__(self, connection_string="mongodb://localhost:27017/"):
        self.client = MongoClient(connection_string)
        self.db = self.client['smart_meter_db']
        
    def create_collections(self):
        """Create all necessary collections with indexes"""
        
        # Drop existing collections to start fresh
        self.db.drop_collection('meter_readings')
        self.db.drop_collection('anomalies')
        self.db.drop_collection('model_performance')
        self.db.drop_collection('stream_stats')
        
        # Collection 1: Meter Readings (merged data)
        meter_readings = self.db['meter_readings']
        meter_readings.create_index([
            ('building_id', ASCENDING),
            ('timestamp', ASCENDING)
        ])
        meter_readings.create_index([('timestamp', DESCENDING)])
        meter_readings.create_index([('anomaly', ASCENDING)])
        print("✓ Created 'meter_readings' collection with indexes")
        
        # Collection 2: Detected Anomalies
        anomalies = self.db['anomalies']
        anomalies.create_index([
            ('building_id', ASCENDING),
            ('timestamp', DESCENDING)
        ])
        anomalies.create_index([('anomaly_score', DESCENDING)])
        print("✓ Created 'anomalies' collection with indexes")
        
        # Collection 3: Model Performance
        model_performance = self.db['model_performance']
        model_performance.create_index([('timestamp', DESCENDING)])
        print("✓ Created 'model_performance' collection with indexes")
        
        # Collection 4: Stream Processing Stats
        stream_stats = self.db['stream_stats']
        stream_stats.create_index([('timestamp', DESCENDING)])
        print("✓ Created 'stream_stats' collection with indexes")
        
        print("\n✅ All collections created successfully!")
        
    def load_training_data(self, train_path, train_features_path):
        """Load LEAD dataset into MongoDB with proper handling of duplicates"""
        
        print("\n📥 Loading LEAD dataset into MongoDB...")
        print("="*60)
        
        # Load train.csv
        print("\n1. Loading train.csv...")
        train_df = pd.read_csv(train_path)
        print(f"   Loaded {len(train_df):,} records")
        
        # Load train_features.csv
        print("\n2. Loading train_features.csv...")
        features_df = pd.read_csv(train_features_path)
        print(f"   Loaded {len(features_df):,} records")
        
        # Smart merge - handle duplicate columns
        print("\n3. Merging datasets...")
        
        # Since both files have the same structure (as seen in your output),
        # we'll use train_features.csv as it has all the features
        merged_df = features_df.copy()
        
        print(f"   Using train_features.csv as primary dataset")
        print(f"   Final dataset: {len(merged_df):,} records with {len(merged_df.columns)} features")
        
        # Convert timestamp
        print("\n4. Converting timestamp...")
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
        print("   ✓ Timestamp converted successfully")
        
        # Insert into MongoDB in batches
        print("\n5. Inserting into MongoDB...")
        batch_size = 5000  # Smaller batches for stability
        total_inserted = 0
        
        for i in range(0, len(merged_df), batch_size):
            batch = merged_df.iloc[i:i+batch_size]
            records = batch.to_dict('records')
            
            # Replace NaN with None for MongoDB
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, (pd.Timestamp, np.datetime64)):
                        record[key] = pd.to_datetime(value).to_pydatetime()
                    elif hasattr(value, 'item'):  # numpy types
                        record[key] = value.item()
            
            self.db['meter_readings'].insert_many(records)
            total_inserted += len(records)
            
            if (i + batch_size) % 25000 == 0:
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
                
            # Building count
            pipeline = [
                {
                    '$group': {
                        '_id': '$building_id',
                        'count': {'$sum': 1}
                    }
                },
                {
                    '$count': 'total_buildings'
                }
            ]
            building_stats = list(self.db['meter_readings'].aggregate(pipeline))
            if building_stats:
                print(f"   Total buildings: {building_stats[0]['total_buildings']:,}")

    def create_indexes_for_analytics(self):
        """Create additional indexes for fast analytics"""
        
        print("\n🔍 Creating analytics indexes...")
        
        meter_readings = self.db['meter_readings']
        
        # Compound indexes for common queries
        meter_readings.create_index([('building_id', ASCENDING), ('anomaly', ASCENDING)])
        meter_readings.create_index([('primary_use', ASCENDING), ('anomaly', ASCENDING)])
        meter_readings.create_index([('hour', ASCENDING), ('building_id', ASCENDING)])
        meter_readings.create_index([('site_id', ASCENDING), ('building_id', ASCENDING)])
        
        print("✓ Analytics indexes created")

if __name__ == "__main__":
    print("🚀 Smart Meter Anomaly Detection - MongoDB Setup (FIXED)")
    print("="*60)
    
    # Check if data files exist
    train_path = 'data/train.csv'
    features_path = 'data/train_features.csv'
    
    if not os.path.exists(train_path):
        print(f"❌ {train_path} not found!")
        exit(1)
    if not os.path.exists(features_path):
        print(f"❌ {features_path} not found!")
        exit(1)
    
    # Initialize
    mongo = MongoDBSetup()
    
    # Create collections and indexes
    mongo.create_collections()
    
    # Load data
    print("\n⚠️  This will load 1.7M+ records into MongoDB (may take 5-10 minutes)")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        mongo.load_training_data(train_path, features_path)
        
        # Create analytics indexes
        mongo.create_indexes_for_analytics()
        
        # Show stats
        mongo.get_collection_stats()
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("   1. Create anomaly detection algorithms")
    print("   2. Build Flask API")
    print("   3. Create dashboard")