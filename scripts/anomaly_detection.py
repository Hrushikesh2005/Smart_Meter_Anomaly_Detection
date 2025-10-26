"""
Simple Anomaly Detection Algorithms
Fast and efficient implementations for real-time detection
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

class SimpleAnomalyDetector:
    def __init__(self, connection_string="mongodb://localhost:27017/"):
        self.client = MongoClient(connection_string)
        self.db = self.client['smart_meter_db']
        self.building_stats = {}  # Cache for building statistics
        
    def load_building_statistics(self, building_id=None):
        """Load statistical data for buildings (mean, std, etc.)"""
        
        print("📊 Loading building statistics...")
        
        if building_id:
            buildings = [building_id]
        else:
            # Get all building IDs
            buildings = self.db['meter_readings'].distinct('building_id')
        
        for bid in buildings:
            # Get non-null meter readings for this building
            pipeline = [
                {'$match': {'building_id': bid, 'meter_reading': {'$ne': None}}},
                {'$group': {
                    '_id': '$building_id',
                    'readings': {'$push': '$meter_reading'},
                    'count': {'$sum': 1},
                    'avg': {'$avg': '$meter_reading'},
                    'min': {'$min': '$meter_reading'},
                    'max': {'$max': '$meter_reading'}
                }}
            ]
            
            result = list(self.db['meter_readings'].aggregate(pipeline))
            
            if result:
                data = result[0]
                readings = data['readings']
                
                # Calculate statistics
                self.building_stats[bid] = {
                    'mean': data['avg'],
                    'std': np.std(readings) if len(readings) > 1 else 0,
                    'min': data['min'],
                    'max': data['max'],
                    'count': data['count'],
                    'median': np.median(readings),
                    'q75': np.percentile(readings, 75),
                    'q25': np.percentile(readings, 25)
                }
        
        print(f"✓ Loaded statistics for {len(self.building_stats)} buildings")
        
    def three_sigma_detection(self, meter_reading, building_id):
        """
        Simple 3-sigma rule anomaly detection
        Fast and effective for most cases
        """
        
        if building_id not in self.building_stats:
            self.load_building_statistics(building_id)
        
        if building_id not in self.building_stats:
            return False, 0.0, "No historical data"
        
        stats = self.building_stats[building_id]
        
        if stats['std'] == 0:
            return False, 0.0, "No variance in data"
        
        # Calculate z-score
        z_score = abs(meter_reading - stats['mean']) / stats['std']
        
        # 3-sigma rule: anomaly if |z| > 3
        is_anomaly = z_score > 3.0
        confidence = min(z_score / 3.0, 1.0)  # Normalize to 0-1
        
        return is_anomaly, confidence, f"Z-score: {z_score:.2f}"
    
    def iqr_detection(self, meter_reading, building_id):
        """
        Interquartile Range (IQR) method
        Robust to outliers
        """
        
        if building_id not in self.building_stats:
            self.load_building_statistics(building_id)
        
        if building_id not in self.building_stats:
            return False, 0.0, "No historical data"
        
        stats = self.building_stats[building_id]
        
        iqr = stats['q75'] - stats['q25']
        
        if iqr == 0:
            return False, 0.0, "No IQR variance"
        
        # IQR rule: anomaly if outside 1.5 * IQR from quartiles
        lower_bound = stats['q25'] - 1.5 * iqr
        upper_bound = stats['q75'] + 1.5 * iqr
        
        is_anomaly = meter_reading < lower_bound or meter_reading > upper_bound
        
        # Calculate confidence based on distance from bounds
        if meter_reading < lower_bound:
            distance = lower_bound - meter_reading
        elif meter_reading > upper_bound:
            distance = meter_reading - upper_bound
        else:
            distance = 0
        
        confidence = min(distance / iqr, 1.0) if iqr > 0 else 0.0
        
        return is_anomaly, confidence, f"IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
    
    def moving_average_detection(self, building_id, window_hours=24, threshold_factor=2.0):
        """
        Moving average anomaly detection
        Detects deviations from recent patterns
        """
        
        # Get recent readings for this building
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours + 1)
        
        pipeline = [
            {
                '$match': {
                    'building_id': building_id,
                    'timestamp': {'$gte': start_time, '$lte': end_time},
                    'meter_reading': {'$ne': None}
                }
            },
            {
                '$sort': {'timestamp': 1}
            },
            {
                '$limit': window_hours + 1
            }
        ]
        
        readings = list(self.db['meter_readings'].aggregate(pipeline))
        
        if len(readings) < window_hours:
            return False, 0.0, f"Insufficient data: {len(readings)} readings"
        
        # Calculate moving average from all but last reading
        historical_readings = [r['meter_reading'] for r in readings[:-1]]
        current_reading = readings[-1]['meter_reading']
        
        moving_avg = np.mean(historical_readings)
        moving_std = np.std(historical_readings)
        
        if moving_std == 0:
            return False, 0.0, "No variance in recent data"
        
        # Check if current reading deviates significantly
        deviation = abs(current_reading - moving_avg)
        threshold = threshold_factor * moving_std
        
        is_anomaly = deviation > threshold
        confidence = min(deviation / threshold, 1.0) if threshold > 0 else 0.0
        
        return is_anomaly, confidence, f"MA: {moving_avg:.2f}, Deviation: {deviation:.2f}"
    
    def ensemble_detection(self, meter_reading, building_id):
        """
        Ensemble method combining multiple algorithms
        More robust and accurate
        """
        
        methods = []
        
        # Run all detection methods
        sigma_result = self.three_sigma_detection(meter_reading, building_id)
        iqr_result = self.iqr_detection(meter_reading, building_id)
        ma_result = self.moving_average_detection(building_id)
        
        methods = [
            ('3-Sigma', sigma_result),
            ('IQR', iqr_result),
            ('Moving Avg', ma_result)
        ]
        
        # Combine results
        anomaly_votes = sum(1 for _, (is_anomaly, _, _) in methods if is_anomaly)
        total_votes = len(methods)
        confidence_scores = [conf for _, (_, conf, _) in methods]
        avg_confidence = np.mean(confidence_scores)
        
        # Decision: majority vote
        is_ensemble_anomaly = anomaly_votes >= 2  # At least 2 out of 3
        
        details = {
            'methods': {name: {'anomaly': anom, 'confidence': conf, 'details': det} 
                       for name, (anom, conf, det) in methods},
            'votes': f"{anomaly_votes}/{total_votes}",
            'avg_confidence': avg_confidence
        }
        
        return is_ensemble_anomaly, avg_confidence, details
    
    def detect_anomaly_batch(self, building_ids=None, method='ensemble'):
        """
        Batch anomaly detection for multiple buildings
        """
        
        if building_ids is None:
            building_ids = self.db['meter_readings'].distinct('building_id')
        
        results = []
        
        print(f"🔍 Running {method} detection on {len(building_ids)} buildings...")
        
        for i, building_id in enumerate(building_ids):
            # Get latest reading for this building
            latest = self.db['meter_readings'].find_one(
                {'building_id': building_id, 'meter_reading': {'$ne': None}},
                sort=[('timestamp', -1)]
            )
            
            if latest and latest['meter_reading'] is not None:
                meter_reading = latest['meter_reading']
                
                if method == '3sigma':
                    is_anomaly, confidence, details = self.three_sigma_detection(meter_reading, building_id)
                elif method == 'iqr':
                    is_anomaly, confidence, details = self.iqr_detection(meter_reading, building_id)
                elif method == 'moving_avg':
                    is_anomaly, confidence, details = self.moving_average_detection(building_id)
                else:  # ensemble
                    is_anomaly, confidence, details = self.ensemble_detection(meter_reading, building_id)
                
                results.append({
                    'building_id': building_id,
                    'meter_reading': meter_reading,
                    'timestamp': latest['timestamp'],
                    'is_anomaly': is_anomaly,
                    'confidence': confidence,
                    'method': method,
                    'details': details
                })
            
            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(building_ids)} buildings")
        
        print(f"✓ Completed detection. Found {sum(1 for r in results if r['is_anomaly'])} anomalies")
        
        return results
    
    def save_anomaly_results(self, results):
        """Save anomaly detection results to MongoDB"""
        
        if not results:
            return
        
        # Prepare documents for insertion
        documents = []
        for result in results:
            if result['is_anomaly']:
                doc = {
                    'building_id': result['building_id'],
                    'timestamp': result['timestamp'],
                    'meter_reading': result['meter_reading'],
                    'anomaly_score': result['confidence'],
                    'detection_method': result['method'],
                    'details': result['details'],
                    'detected_at': datetime.now()
                }
                documents.append(doc)
        
        if documents:
            self.db['anomalies'].insert_many(documents)
            print(f"✓ Saved {len(documents)} anomaly records to database")

def demo_anomaly_detection():
    """Demonstration of anomaly detection algorithms"""
    
    print("🎯 Smart Meter Anomaly Detection Demo")
    print("="*50)
    
    detector = SimpleAnomalyDetector()
    
    # Load statistics for first 10 buildings
    detector.load_building_statistics()
    
    # Run ensemble detection on first 20 buildings
    building_ids = list(range(1, 21))
    results = detector.detect_anomaly_batch(building_ids, method='ensemble')
    
    # Display results
    print(f"\n📊 Detection Results:")
    print("-" * 80)
    print(f"{'Building':<10} {'Reading':<12} {'Anomaly':<8} {'Confidence':<12} {'Details'}")
    print("-" * 80)
    
    for result in results[:15]:  # Show first 15
        building_id = result['building_id']
        reading = result['meter_reading']
        is_anomaly = "YES" if result['is_anomaly'] else "NO"
        confidence = f"{result['confidence']:.3f}"
        
        if isinstance(result['details'], dict) and 'votes' in result['details']:
            details = result['details']['votes']
        else:
            details = str(result['details'])[:30]
        
        print(f"{building_id:<10} {reading:<12.2f} {is_anomaly:<8} {confidence:<12} {details}")
    
    # Save anomalies to database
    detector.save_anomaly_results(results)
    
    print(f"\n✅ Demo completed!")

if __name__ == "__main__":
    demo_anomaly_detection()