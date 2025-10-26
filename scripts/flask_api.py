"""
Simple Flask API for Smart Meter Anomaly Detection
Fast and lightweight REST endpoints
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime, timedelta
import json
from bson import ObjectId
import numpy as np

# Import our anomaly detection module
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from anomaly_detection import SimpleAnomalyDetector

# Custom JSON encoder for MongoDB ObjectId and datetime
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend
app.json_encoder = CustomJSONEncoder

# Initialize MongoDB connection and anomaly detector
client = MongoClient('mongodb://localhost:27017/')
db = client['smart_meter_db']
detector = SimpleAnomalyDetector()

@app.route('/')
def home():
    """API welcome message"""
    return jsonify({
        'message': 'Smart Meter Anomaly Detection API',
        'version': '1.0',
        'endpoints': [
            '/api/stats',
            '/api/buildings',
            '/api/building/<building_id>',
            '/api/anomalies',
            '/api/detect/<building_id>',
            '/api/recent-readings'
        ]
    })

@app.route('/api/stats')
def get_stats():
    """Get overall database statistics"""
    
    # Total records
    total_readings = db['meter_readings'].count_documents({})
    
    # Total buildings
    total_buildings = len(db['meter_readings'].distinct('building_id'))
    
    # Anomaly count
    anomaly_count = db['meter_readings'].count_documents({'anomaly': 1})
    
    # Recent anomalies detected by our system
    recent_anomalies = db['anomalies'].count_documents({})
    
    # Building types
    building_types = list(db['meter_readings'].aggregate([
        {'$group': {'_id': '$primary_use', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}},
        {'$limit': 10}
    ]))
    
    return jsonify({
        'total_readings': total_readings,
        'total_buildings': total_buildings,
        'labeled_anomalies': anomaly_count,
        'detected_anomalies': recent_anomalies,
        'anomaly_rate': round((anomaly_count / total_readings) * 100, 2) if total_readings > 0 else 0,
        'building_types': building_types,
        'last_updated': datetime.now().isoformat()
    })

@app.route('/api/buildings')
def get_buildings():
    """Get list of all buildings with basic info"""
    
    limit = request.args.get('limit', 50, type=int)
    page = request.args.get('page', 1, type=int)
    skip = (page - 1) * limit
    
    pipeline = [
        {
            '$group': {
                '_id': '$building_id',
                'primary_use': {'$first': '$primary_use'},
                'site_id': {'$first': '$site_id'},
                'square_feet': {'$first': '$square_feet'},
                'floor_count': {'$first': '$floor_count'},
                'total_readings': {'$sum': 1},
                'anomaly_count': {'$sum': {'$cond': [{'$eq': ['$anomaly', 1]}, 1, 0]}},
                'last_reading': {'$max': '$timestamp'},
                'avg_consumption': {'$avg': '$meter_reading'}
            }
        },
        {
            '$addFields': {
                'anomaly_rate': {
                    '$multiply': [
                        {'$divide': ['$anomaly_count', '$total_readings']},
                        100
                    ]
                }
            }
        },
        {'$sort': {'_id': 1}},
        {'$skip': skip},
        {'$limit': limit}
    ]
    
    buildings = list(db['meter_readings'].aggregate(pipeline))
    
    # Rename _id to building_id for clarity
    for building in buildings:
        building['building_id'] = building.pop('_id')
    
    return jsonify({
        'buildings': buildings,
        'page': page,
        'limit': limit,
        'total_buildings': len(db['meter_readings'].distinct('building_id'))
    })

@app.route('/api/building/<int:building_id>')
def get_building_details(building_id):
    """Get detailed information for a specific building"""
    
    # Building basic info
    building_info = db['meter_readings'].find_one(
        {'building_id': building_id},
        {
            'building_id': 1, 'primary_use': 1, 'site_id': 1,
            'square_feet': 1, 'floor_count': 1, 'year_built': 1
        }
    )
    
    if not building_info:
        return jsonify({'error': 'Building not found'}), 404
    
    # Recent readings (last 24 hours worth)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    recent_readings = list(db['meter_readings'].find(
        {
            'building_id': building_id,
            'timestamp': {'$gte': start_time, '$lte': end_time}
        },
        {
            'timestamp': 1, 'meter_reading': 1, 'anomaly': 1,
            'air_temperature': 1, 'hour': 1
        }
    ).sort('timestamp', -1).limit(24))
    
    # Statistics
    stats_pipeline = [
        {'$match': {'building_id': building_id, 'meter_reading': {'$ne': None}}},
        {
            '$group': {
                '_id': None,
                'avg_consumption': {'$avg': '$meter_reading'},
                'min_consumption': {'$min': '$meter_reading'},
                'max_consumption': {'$max': '$meter_reading'},
                'total_readings': {'$sum': 1},
                'anomaly_count': {'$sum': {'$cond': [{'$eq': ['$anomaly', 1]}, 1, 0]}}
            }
        }
    ]
    
    stats = list(db['meter_readings'].aggregate(stats_pipeline))
    building_stats = stats[0] if stats else {}
    
    return jsonify({
        'building_info': building_info,
        'recent_readings': recent_readings,
        'statistics': building_stats
    })

@app.route('/api/anomalies')
def get_anomalies():
    """Get recent anomalies detected by the system"""
    
    limit = request.args.get('limit', 20, type=int)
    hours = request.args.get('hours', 24, type=int)
    
    # Get anomalies from the last N hours
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    anomalies = list(db['anomalies'].find(
        {'detected_at': {'$gte': start_time, '$lte': end_time}}
    ).sort('detected_at', -1).limit(limit))
    
    # Also get labeled anomalies from the dataset
    labeled_anomalies = list(db['meter_readings'].find(
        {
            'anomaly': 1,
            'timestamp': {'$gte': start_time, '$lte': end_time}
        },
        {
            'building_id': 1, 'timestamp': 1, 'meter_reading': 1,
            'primary_use': 1, 'air_temperature': 1
        }
    ).sort('timestamp', -1).limit(limit))
    
    return jsonify({
        'detected_anomalies': anomalies,
        'labeled_anomalies': labeled_anomalies,
        'time_range_hours': hours
    })

@app.route('/api/detect/<int:building_id>')
def detect_anomaly(building_id):
    """Run anomaly detection on a specific building"""
    
    method = request.args.get('method', 'ensemble')
    
    # Get latest reading for the building
    latest_reading = db['meter_readings'].find_one(
        {'building_id': building_id, 'meter_reading': {'$ne': None}},
        sort=[('timestamp', -1)]
    )
    
    if not latest_reading:
        return jsonify({'error': 'No readings found for building'}), 404
    
    meter_reading = latest_reading['meter_reading']
    
    # Run detection based on method
    if method == '3sigma':
        is_anomaly, confidence, details = detector.three_sigma_detection(meter_reading, building_id)
    elif method == 'iqr':
        is_anomaly, confidence, details = detector.iqr_detection(meter_reading, building_id)
    elif method == 'moving_avg':
        is_anomaly, confidence, details = detector.moving_average_detection(building_id)
    else:  # ensemble
        is_anomaly, confidence, details = detector.ensemble_detection(meter_reading, building_id)
    
    result = {
        'building_id': building_id,
        'meter_reading': meter_reading,
        'timestamp': latest_reading['timestamp'],
        'is_anomaly': is_anomaly,
        'confidence': confidence,
        'method': method,
        'details': details,
        'building_info': {
            'primary_use': latest_reading.get('primary_use'),
            'site_id': latest_reading.get('site_id')
        }
    }
    
    # Save to database if anomaly detected
    if is_anomaly:
        anomaly_doc = {
            'building_id': building_id,
            'timestamp': latest_reading['timestamp'],
            'meter_reading': meter_reading,
            'anomaly_score': confidence,
            'detection_method': method,
            'details': details,
            'detected_at': datetime.now()
        }
        db['anomalies'].insert_one(anomaly_doc)
    
    return jsonify(result)

@app.route('/api/recent-readings')
def get_recent_readings():
    """Get recent meter readings across all buildings"""
    
    limit = request.args.get('limit', 100, type=int)
    hours = request.args.get('hours', 1, type=int)
    
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    readings = list(db['meter_readings'].find(
        {
            'timestamp': {'$gte': start_time, '$lte': end_time},
            'meter_reading': {'$ne': None}
        },
        {
            'building_id': 1, 'timestamp': 1, 'meter_reading': 1,
            'anomaly': 1, 'primary_use': 1, 'air_temperature': 1
        }
    ).sort('timestamp', -1).limit(limit))
    
    return jsonify({
        'readings': readings,
        'count': len(readings),
        'time_range_hours': hours
    })

@app.route('/api/batch-detect')
def batch_detect():
    """Run anomaly detection on multiple buildings"""
    
    method = request.args.get('method', 'ensemble')
    building_count = request.args.get('buildings', 20, type=int)
    
    # Get list of building IDs
    all_buildings = db['meter_readings'].distinct('building_id')
    building_ids = all_buildings[:building_count]
    
    # Run batch detection
    results = detector.detect_anomaly_batch(building_ids, method=method)
    
    # Save results
    detector.save_anomaly_results(results)
    
    # Summary
    anomaly_count = sum(1 for r in results if r['is_anomaly'])
    
    return jsonify({
        'results': results,
        'summary': {
            'total_buildings': len(results),
            'anomalies_found': anomaly_count,
            'method': method,
            'detection_time': datetime.now().isoformat()
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Smart Meter Anomaly Detection API...")
    print("📊 Loading building statistics...")
    
    # Pre-load statistics for faster responses
    detector.load_building_statistics()
    
    print("✅ API ready!")
    print("🌐 Available endpoints:")
    print("   http://localhost:5000/")
    print("   http://localhost:5000/api/stats")
    print("   http://localhost:5000/api/buildings")
    print("   http://localhost:5000/api/anomalies")
    print("   http://localhost:5000/api/detect/1")
    print("   http://localhost:5000/api/batch-detect")
    
    app.run(debug=True, host='0.0.0.0', port=5000)