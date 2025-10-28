"""
Simple XGBoost Classification Model for Anomaly Detection
Trains a model on the dataset and saves it for predictions
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

class AnomalyModelTrainer:
    def __init__(self):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['smart_meter_db']
        self.model = None
        self.feature_columns = None
        
    def load_data(self, sample_size=700000):
        """Load data from MongoDB for training"""
        print(f"📥 Loading {sample_size:,} samples from MongoDB...")
        
        # Sample data from database
        pipeline = [
            {'$match': {'meter_reading': {'$ne': None}}},
            {'$sample': {'size': sample_size}}
        ]
        
        data = list(self.db['meter_readings'].aggregate(pipeline))
        df = pd.DataFrame(data)
        
        print(f"✅ Loaded {len(df):,} records")
        return df
    
    def prepare_features(self, df):
        """Select and prepare features for training"""
        print("\n🔧 Preparing features...")
        
        # Select numeric features only
        feature_columns = [
            'meter_reading', 'air_temperature', 'wind_speed',
            'hour', 'weekday', 'month', 'square_feet', 'floor_count',
            'cloud_coverage', 'dew_temperature', 'sea_level_pressure'
        ]
        
        # Keep only available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Remove rows with NaN in features or target
        df_clean = df[available_features + ['anomaly']].dropna()
        
        X = df_clean[available_features]
        y = df_clean['anomaly']
        
        self.feature_columns = available_features
        
        print(f"✅ Using {len(available_features)} features: {available_features}")
        print(f"✅ Clean dataset: {len(X):,} samples")
        print(f"   - Normal (0): {(y == 0).sum():,}")
        print(f"   - Anomaly (1): {(y == 1).sum():,}")
        
        return X, y
    
    def train_model(self, X, y):
        """Train XGBoost classifier"""
        print("\n🤖 Training XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model trained successfully!")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Print detailed metrics
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0][0]:,}")
        print(f"   False Positives: {cm[0][1]:,}")
        print(f"   False Negatives: {cm[1][0]:,}")
        print(f"   True Positives:  {cm[1][1]:,}")
        
        return accuracy
    
    def save_model(self, filename='xgboost_model.pkl'):
        """Save trained model and feature columns"""
        print(f"\n💾 Saving model to {filename}...")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved successfully!")
        print(f"   Features: {self.feature_columns}")
    
    def run_training(self, sample_size=700000):
        """Complete training pipeline"""
        print("\n" + "="*60)
        print("  🤖 XGBoost Anomaly Detection Model Training")
        print("="*60)
        
        # Load data
        df = self.load_data(sample_size)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Train model
        accuracy = self.train_model(X, y)
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("  ✅ Training Complete!")
        print("="*60)
        print(f"  Model Accuracy: {accuracy*100:.2f}%")
        print(f"  Model saved: xgboost_model.pkl")
        print(f"  Ready for predictions!")
        print("="*60 + "\n")
        
        return accuracy

if __name__ == "__main__":
    try:
        trainer = AnomalyModelTrainer()
        trainer.run_training(sample_size=700000)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
