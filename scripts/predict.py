"""
Simple Prediction Service using trained XGBoost model
"""

import pickle
import pandas as pd
import numpy as np

class AnomalyPredictor:
    def __init__(self, model_path='xgboost_model.pkl'):
        """Load trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.feature_columns = model_data['feature_columns']
            print(f"✅ Model loaded from {model_path}")
        except FileNotFoundError:
            raise Exception(f"Model file not found: {model_path}. Please train the model first!")
    
    def predict(self, input_data):
        """
        Make prediction from input dictionary
        
        Args:
            input_data: dict with feature names and values
            
        Returns:
            dict with prediction and probability
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
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'Anomaly' if prediction == 1 else 'Normal',
            'confidence': float(probability[int(prediction)]),
            'probability_normal': float(probability[0]),
            'probability_anomaly': float(probability[1])
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
