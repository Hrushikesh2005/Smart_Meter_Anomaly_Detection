"""
Simple Prediction Service using trained XGBoost model
Includes threshold-based analysis to identify anomalous parameters
"""

import pickle
import pandas as pd
import numpy as np
from pymongo import MongoClient

class AnomalyPredictor:
    def __init__(self, model_path='xgboost_model.pkl'):
        """Load trained model and calculate thresholds from dataset"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.feature_columns = model_data['feature_columns']
            print(f"✅ Model loaded from {model_path}")
            
            # Calculate thresholds from dataset
            self._calculate_thresholds()
            
        except FileNotFoundError:
            raise Exception(f"Model file not found: {model_path}. Please train the model first!")
    
    def _calculate_thresholds(self):
        """Calculate statistical thresholds from dataset"""
        try:
            client = MongoClient('mongodb://localhost:27017/')
            db = client['smart_meter_db']
            
            # Sample data for threshold calculation
            pipeline = [
                {'$match': {'meter_reading': {'$ne': None}}},
                {'$sample': {'size': 10000}}
            ]
            
            data = list(db['meter_readings'].aggregate(pipeline))
            df = pd.DataFrame(data)
            
            # Calculate mean and std for each feature
            self.thresholds = {}
            for col in self.feature_columns:
                if col in df.columns:
                    values = df[col].dropna()
                    mean = values.mean()
                    std = values.std()
                    
                    # Use 3-sigma rule (99.7% of data falls within 3 std)
                    self.thresholds[col] = {
                        'mean': mean,
                        'std': std,
                        'min': mean - 3 * std,
                        'max': mean + 3 * std,
                        'q1': values.quantile(0.25),
                        'q3': values.quantile(0.75)
                    }
            
            print(f"✅ Thresholds calculated for {len(self.thresholds)} features")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not calculate thresholds from DB: {e}")
            # Fallback to default thresholds
            self.thresholds = self._get_default_thresholds()
    
    def _get_default_thresholds(self):
        """Fallback thresholds based on typical ranges"""
        return {
            'meter_reading': {'mean': 150, 'std': 100, 'min': 0, 'max': 500, 'q1': 50, 'q3': 250},
            'air_temperature': {'mean': 20, 'std': 10, 'min': -10, 'max': 50, 'q1': 15, 'q3': 25},
            'wind_speed': {'mean': 5, 'std': 3, 'min': 0, 'max': 15, 'q1': 2, 'q3': 8},
            'hour': {'mean': 12, 'std': 7, 'min': 0, 'max': 23, 'q1': 6, 'q3': 18},
            'weekday': {'mean': 3, 'std': 2, 'min': 0, 'max': 6, 'q1': 1, 'q3': 5},
            'month': {'mean': 6, 'std': 3, 'min': 1, 'max': 12, 'q1': 3, 'q3': 9},
            'square_feet': {'mean': 50000, 'std': 50000, 'min': 1000, 'max': 200000, 'q1': 20000, 'q3': 80000},
            'floor_count': {'mean': 3, 'std': 2, 'min': 1, 'max': 10, 'q1': 1, 'q3': 5},
            'cloud_coverage': {'mean': 4, 'std': 3, 'min': 0, 'max': 9, 'q1': 2, 'q3': 7},
            'dew_temperature': {'mean': 15, 'std': 8, 'min': -5, 'max': 30, 'q1': 10, 'q3': 20},
            'sea_level_pressure': {'mean': 1013, 'std': 10, 'min': 990, 'max': 1040, 'q1': 1008, 'q3': 1018}
        }
    
    def _check_anomalous_parameters(self, input_data):
        """Identify which parameters are anomalous based on thresholds"""
        anomalous_params = []
        
        for param, value in input_data.items():
            if param not in self.thresholds:
                continue
            
            thresh = self.thresholds[param]
            
            # Check if value is outside 3-sigma range
            if value < thresh['min'] or value > thresh['max']:
                deviation = abs(value - thresh['mean']) / thresh['std'] if thresh['std'] > 0 else 0
                
                anomalous_params.append({
                    'parameter': param,
                    'value': value,
                    'expected_range': f"{thresh['min']:.2f} - {thresh['max']:.2f}",
                    'mean': thresh['mean'],
                    'deviation': deviation,
                    'severity': 'HIGH' if deviation > 4 else 'MEDIUM' if deviation > 3 else 'LOW'
                })
        
        return anomalous_params
    
    def predict(self, input_data):
        """
        Make prediction from input dictionary
        
        Args:
            input_data: dict with feature names and values
            
        Returns:
            dict with prediction, probability, and anomalous parameters
        """
        # Create DataFrame with all required features
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present (fill missing with 0)
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select only required features in correct order
        X = input_df[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Check which parameters are anomalous
        anomalous_params = self._check_anomalous_parameters(input_data)
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'Anomaly' if prediction == 1 else 'Normal',
            'confidence': float(probability[int(prediction)]),
            'probability_normal': float(probability[0]),
            'probability_anomaly': float(probability[1]),
            'anomalous_parameters': anomalous_params,
            'anomaly_count': len(anomalous_params)
        }
    
    def get_feature_list(self):
        """Return list of required features"""
        return self.feature_columns

if __name__ == "__main__":
    # Test prediction
    predictor = AnomalyPredictor()
    
    # Sample input
    sample_input = {
        'meter_reading': 150.5,
        'air_temperature': 22.5,
        'wind_speed': 5.2,
        'hour': 14,
        'weekday': 3,
        'month': 6,
        'square_feet': 5000,
        'floor_count': 3,
        'cloud_coverage': 4,
        'dew_temperature': 15.0,
        'sea_level_pressure': 1013.25
    }
    
    result = predictor.predict(sample_input)
    print(f"\n🔮 Prediction Result:")
    print(f"   Status: {result['prediction_label']}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    print(f"   Prob(Normal): {result['probability_normal']*100:.2f}%")
    print(f"   Prob(Anomaly): {result['probability_anomaly']*100:.2f}%")
    
    if result['anomalous_parameters']:
        print(f"\n⚠️  Anomalous Parameters Detected ({result['anomaly_count']}):")
        for param in result['anomalous_parameters']:
            print(f"   • {param['parameter']}: {param['value']}")
            print(f"     Expected: {param['expected_range']}")
            print(f"     Severity: {param['severity']} (σ={param['deviation']:.2f})")
    else:
        print(f"\n✅ All parameters within normal ranges")

