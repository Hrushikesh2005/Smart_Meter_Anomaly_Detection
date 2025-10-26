"""
Simple Flask API for Smart Meter Anomaly Detection - Optimized Version
Fast and lightweight REST endpoints with better error handling
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime, timedelta
import json
from bson import ObjectId
import numpy as np

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

# Global variables for database connection and caching
client = None
db = None
cache = {}  # Simple in-memory cache
cache_timeout = 300  # 5 minutes

def get_db_connection():
    """Get MongoDB connection with error handling and optimization"""
    global client, db
    if client is None:
        try:
            client = MongoClient(
                'mongodb://localhost:27017/', 
                serverSelectionTimeoutMS=5000,
                maxPoolSize=10,  # Connection pooling for performance
                retryWrites=True
            )
            db = client['smart_meter_db']
            # Test connection
            client.server_info()
            print("✅ MongoDB connected successfully")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            client = None
            db = None
    return db

def get_cached_data(key, fetch_function, *args, **kwargs):
    """Simple caching mechanism for performance optimization"""
    import time
    
    current_time = time.time()
    
    # Check if we have cached data and it's not expired
    if key in cache:
        cached_data, timestamp = cache[key]
        if current_time - timestamp < cache_timeout:
            return cached_data
    
    # Fetch fresh data
    data = fetch_function(*args, **kwargs)
    cache[key] = (data, current_time)
    return data

@app.route('/')
def home():
    """API welcome message"""
    return jsonify({
        'message': 'Smart Meter Anomaly Detection API - Phase 3',
        'version': '3.0',
        'status': 'running',
        'endpoints': [
            '/api/stats',
            '/api/buildings',
            '/api/building/<building_id>',
            '/api/anomalies',
            '/api/detect/<building_id>',
            '/api/recent-readings',
            '/api/weather-analysis',
            '/api/stream-mining-demo',
            '/api/performance-stats'
        ],
        'features': [
            'Real-time building monitoring',
            'Anomaly detection algorithms', 
            'Weather correlation analysis',
            'Stream mining demos (Bloom Filter + DGIM)',
            'Performance optimization'
        ]
    })

@app.route('/api/stats')
def get_stats():
    """Get overall database statistics"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        # Total records
        total_readings = db['meter_readings'].count_documents({})
        
        # Total buildings
        total_buildings = len(db['meter_readings'].distinct('building_id'))
        
        # Anomaly count from dataset
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
    
    except Exception as e:
        return jsonify({'error': f'Database query failed: {str(e)}'}), 500

@app.route('/api/buildings')
def get_buildings():
    """Get list of all buildings with basic info"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        limit = request.args.get('limit', 20, type=int)
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
    
    except Exception as e:
        return jsonify({'error': f'Database query failed: {str(e)}'}), 500

@app.route('/api/building/<int:building_id>')
def get_building_details(building_id):
    """Get detailed information for a specific building"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
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
        
        # Recent readings (last 24 records)
        recent_readings = list(db['meter_readings'].find(
            {'building_id': building_id, 'meter_reading': {'$ne': None}},
            {
                'timestamp': 1, 'meter_reading': 1, 'anomaly': 1,
                'air_temperature': 1, 'hour': 1
            }
        ).sort('timestamp', -1).limit(24))
        
        # Basic statistics
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
    
    except Exception as e:
        return jsonify({'error': f'Database query failed: {str(e)}'}), 500

@app.route('/api/detect/<int:building_id>')
def simple_detect_anomaly(building_id):
    """Simple anomaly detection using statistical methods"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        # Get latest reading for the building
        latest_reading = db['meter_readings'].find_one(
            {'building_id': building_id, 'meter_reading': {'$ne': None}},
            sort=[('timestamp', -1)]
        )
        
        if not latest_reading:
            return jsonify({'error': 'No readings found for building'}), 404
        
        meter_reading = latest_reading['meter_reading']
        
        # Get historical statistics for this building
        stats_pipeline = [
            {'$match': {'building_id': building_id, 'meter_reading': {'$ne': None}}},
            {
                '$group': {
                    '_id': None,
                    'mean': {'$avg': '$meter_reading'},
                    'count': {'$sum': 1},
                    'readings': {'$push': '$meter_reading'}
                }
            }
        ]
        
        stats_result = list(db['meter_readings'].aggregate(stats_pipeline))
        
        if not stats_result:
            return jsonify({'error': 'No historical data for building'}), 404
        
        stats = stats_result[0]
        readings = stats['readings']
        mean = stats['mean']
        std = np.std(readings) if len(readings) > 1 else 0
        
        # Simple 3-sigma detection
        if std > 0:
            z_score = abs(meter_reading - mean) / std
            is_anomaly = z_score > 3.0
            confidence = min(z_score / 3.0, 1.0)
        else:
            is_anomaly = False
            confidence = 0.0
            z_score = 0.0
        
        result = {
            'building_id': building_id,
            'meter_reading': meter_reading,
            'timestamp': latest_reading['timestamp'],
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'z_score': z_score,
            'method': '3-sigma',
            'building_info': {
                'primary_use': latest_reading.get('primary_use'),
                'site_id': latest_reading.get('site_id')
            },
            'statistics': {
                'mean': mean,
                'std': std,
                'total_readings': stats['count']
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/api/anomalies')
def get_anomalies():
    """Get recent anomalies from the dataset"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        limit = request.args.get('limit', 20, type=int)
        
        # Get labeled anomalies from the dataset
        labeled_anomalies = list(db['meter_readings'].find(
            {'anomaly': 1, 'meter_reading': {'$ne': None}},
            {
                'building_id': 1, 'timestamp': 1, 'meter_reading': 1,
                'primary_use': 1, 'air_temperature': 1, 'site_id': 1
            }
        ).sort('timestamp', -1).limit(limit))
        
        # Get detected anomalies from our system
        detected_anomalies = list(db['anomalies'].find().sort('detected_at', -1).limit(limit))
        
        return jsonify({
            'labeled_anomalies': labeled_anomalies,
            'detected_anomalies': detected_anomalies,
            'count': {
                'labeled': len(labeled_anomalies),
                'detected': len(detected_anomalies)
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Database query failed: {str(e)}'}), 500

@app.route('/api/weather-analysis')
def get_weather_analysis():
    """Get weather correlation analysis"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        # Get sample data for weather analysis (cached approach for performance)
        pipeline = [
            {'$match': {'air_temperature': {'$ne': None}, 'meter_reading': {'$ne': None}}},
            {'$sample': {'size': 2000}},
            {'$project': {
                'meter_reading': 1, 'air_temperature': 1, 'wind_speed': 1,
                'primary_use': 1, 'building_id': 1, 'timestamp': 1
            }}
        ]
        
        data = list(db['meter_readings'].aggregate(pipeline))
        
        if len(data) < 100:
            return jsonify({'error': 'Insufficient weather data'}), 404
        
        # Calculate correlations using numpy for performance
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Temperature correlation
        temp_corr = df['meter_reading'].corr(df['air_temperature'])
        wind_corr = df['meter_reading'].corr(df['wind_speed']) if 'wind_speed' in df.columns else 0
        
        # Temperature ranges analysis
        df['temp_range'] = pd.cut(df['air_temperature'], 
                                 bins=5, labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'])
        temp_analysis = df.groupby('temp_range')['meter_reading'].agg(['mean', 'std', 'count'])
        
        # Convert to serializable format
        temp_ranges = {}
        for temp_range in temp_analysis.index:
            if pd.notna(temp_analysis.loc[temp_range, 'mean']):
                temp_ranges[str(temp_range)] = {
                    'mean': float(temp_analysis.loc[temp_range, 'mean']),
                    'std': float(temp_analysis.loc[temp_range, 'std']),
                    'count': int(temp_analysis.loc[temp_range, 'count'])
                }
        
        return jsonify({
            'correlations': {
                'temperature': float(temp_corr) if not pd.isna(temp_corr) else 0,
                'wind_speed': float(wind_corr) if not pd.isna(wind_corr) else 0
            },
            'temperature_analysis': {
                'ranges': temp_ranges,
                'sample_size': len(data)
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Weather analysis failed: {str(e)}'}), 500

@app.route('/api/stream-mining-demo')
def stream_mining_demo():
    """Demonstrate stream mining capabilities with real data"""
    
    try:
        # Import stream mining classes
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from stream_mining import BloomFilter, DGIM
        
        db = get_db_connection()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Initialize stream mining components with optimized parameters
        bloom_filter = BloomFilter(capacity=1000, error_rate=0.01)
        dgim = DGIM(window_size=3600)  # 1 hour window
        
        # Get recent anomalies for demonstration
        anomalies = list(db['meter_readings'].find(
            {'anomaly': 1, 'meter_reading': {'$ne': None}},
            {'building_id': 1, 'timestamp': 1, 'meter_reading': 1}
        ).limit(100))
        
        processed_count = 0
        duplicate_count = 0
        current_time = datetime.now().timestamp()
        
        for anomaly in anomalies:
            anomaly_id = f"{anomaly['building_id']}_{anomaly['meter_reading']:.2f}"
            
            # Check for duplicates with Bloom Filter
            if bloom_filter.contains(anomaly_id):
                duplicate_count += 1
            else:
                bloom_filter.add(anomaly_id)
            
            # Add to DGIM for windowed counting
            dgim.add_bit(1, current_time - processed_count * 10)  # Simulate time intervals
            processed_count += 1
        
        # Get statistics
        bf_stats = bloom_filter.get_stats()
        dgim_stats = dgim.get_stats()
        
        return jsonify({
            'stream_mining_results': {
                'total_processed': processed_count,
                'duplicates_detected': duplicate_count,
                'duplicate_rate': (duplicate_count / processed_count * 100) if processed_count > 0 else 0,
                'bloom_filter_stats': {
                    'capacity': bf_stats['capacity'],
                    'items_added': bf_stats['items_added'],
                    'utilization': bf_stats['utilization'] * 100,
                    'estimated_error_rate': bf_stats['estimated_fpr'] * 100
                },
                'dgim_stats': {
                    'window_size': dgim_stats['window_size'],
                    'estimated_count': dgim_stats['estimated_ones_in_window'],
                    'bucket_count': dgim_stats['bucket_count'],
                    'memory_usage': dgim_stats['memory_usage_bytes']
                }
            },
            'demo_timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Stream mining demo failed: {str(e)}'}), 500

@app.route('/api/performance-stats')
def get_performance_stats():
    """Get system performance statistics"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        import time
        start_time = time.time()
        
        # Quick database stats
        total_readings = db['meter_readings'].estimated_document_count()
        total_buildings = len(db['meter_readings'].distinct('building_id'))
        
        # Memory usage estimation (fallback if psutil not available)
        memory_usage = 0
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            memory_usage = 0  # Fallback if psutil not installed
        
        query_time = time.time() - start_time
        
        return jsonify({
            'performance': {
                'total_readings': total_readings,
                'total_buildings': total_buildings,
                'query_time_ms': round(query_time * 1000, 2),
                'memory_usage_mb': round(memory_usage, 2),
                'api_status': 'optimized'
            },
            'database': {
                'connection_status': 'active',
                'response_time_ms': round(query_time * 1000, 2)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Performance stats failed: {str(e)}'}), 500

@app.route('/api/recent-readings')
def get_recent_readings():
    """Get recent meter readings across all buildings"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        
        readings = list(db['meter_readings'].find(
            {'meter_reading': {'$ne': None}},
            {
                'building_id': 1, 'timestamp': 1, 'meter_reading': 1,
                'anomaly': 1, 'primary_use': 1, 'air_temperature': 1
            }
        ).sort('timestamp', -1).limit(limit))
        
        return jsonify({
            'readings': readings,
            'count': len(readings)
        })
    
    except Exception as e:
        return jsonify({'error': f'Database query failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Smart Meter Anomaly Detection API...")
    
    # Test database connection
    db = get_db_connection()
    if db is not None:
        print("✅ API ready!")
        print("🌐 Available endpoints:")
        print("   http://localhost:5000/")
        print("   http://localhost:5000/api/stats")
        print("   http://localhost:5000/api/buildings")
        print("   http://localhost:5000/api/anomalies")
        print("   http://localhost:5000/api/detect/1")
        
        # Disable debug mode to prevent restart issues
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start API - MongoDB connection failed")
        print("💡 Make sure MongoDB is running on localhost:27017")