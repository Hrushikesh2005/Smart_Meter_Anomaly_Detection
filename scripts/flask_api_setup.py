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

def safe_round(value, decimals=2):
    """Safely round a numeric value, handling None values"""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return None

def safe_timestamp(timestamp):
    """Safely format timestamp, handling both datetime objects and strings"""
    if timestamp is None:
        return None
    if isinstance(timestamp, str):
        return timestamp
    try:
        return timestamp.isoformat()
    except AttributeError:
        return str(timestamp)

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

# Import ML API
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ml_api import ml_api
    print("✅ ML API loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load ML API: {e}")
    ml_api = None

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

@app.route('/api/meter-readings', methods=['POST'])
def add_meter_reading():
    """Add a new meter reading to the database"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['building_id', 'meter_reading', 'timestamp']
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare the meter reading document
        meter_reading = {
            'building_id': int(data['building_id']),
            'meter_reading': float(data['meter_reading']),
            'timestamp': data['timestamp'],
            'anomaly': int(data.get('anomaly', 0)),
            'air_temperature': float(data['air_temperature']) if data.get('air_temperature') else None,
            'wind_speed': float(data['wind_speed']) if data.get('wind_speed') else None,
            'primary_use': data.get('primary_use', 'Unknown'),
            'meter': 0,  # Default electricity meter
            'site_id': data['building_id'],  # Use building_id as site_id
            'created_at': datetime.now().isoformat(),
            # Add some default values for compatibility
            'square_feet': 5000,  # Default value
            'floor_count': 3,     # Default value
            'year_built': 2010    # Default value
        }
        
        # Insert into database
        result = db['meter_readings'].insert_one(meter_reading)
        
        # Update cache if it exists (invalidate cached stats)
        global cached_stats, cache_timestamp
        cached_stats = None
        cache_timestamp = 0
        
        return jsonify({
            'status': 'success',
            'message': 'Meter reading added successfully',
            'inserted_id': str(result.inserted_id),
            'building_id': meter_reading['building_id'],
            'meter_reading': meter_reading['meter_reading']
        }), 201
        
    except ValueError as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        print(f"Error adding meter reading: {e}")
        return jsonify({'error': f'Failed to add meter reading: {str(e)}'}), 500

@app.route('/api/weather-analysis')
def get_weather_analysis():
    """Get weather correlation analysis with scatter plot data"""
    
    db = get_db_connection()
    if db is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        # Get sample data for weather analysis (cached approach for performance)
        pipeline = [
            {'$match': {'air_temperature': {'$ne': None}, 'meter_reading': {'$ne': None}}},
            {'$sample': {'size': 500}},  # Reduced for better performance in scatter plot
            {'$project': {
                'meter_reading': 1, 'air_temperature': 1, 'wind_speed': 1,
                'primary_use': 1, 'building_id': 1, 'timestamp': 1
            }}
        ]
        
        data = list(db['meter_readings'].aggregate(pipeline))
        
        if len(data) < 50:
            return jsonify({'error': 'Insufficient weather data'}), 404
        
        # Calculate correlations using pandas
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(data)
        
        # Temperature correlation and scatter data
        temp_corr = df['meter_reading'].corr(df['air_temperature'])
        wind_corr = df['meter_reading'].corr(df['wind_speed']) if 'wind_speed' in df.columns else 0
        
        # Prepare scatter plot data
        scatter_data = {
            'temperature': [
                {'x': float(row['air_temperature']), 'y': float(row['meter_reading'])}
                for _, row in df.iterrows() 
                if pd.notna(row['air_temperature']) and pd.notna(row['meter_reading'])
            ][:100],  # Limit to 100 points for performance
            'wind_speed': [
                {'x': float(row['wind_speed']), 'y': float(row['meter_reading'])}
                for _, row in df.iterrows() 
                if pd.notna(row.get('wind_speed', 0)) and pd.notna(row['meter_reading'])
            ][:100] if 'wind_speed' in df.columns else []
        }
        
        # Calculate trend lines using linear regression
        def calculate_trend_line(x_vals, y_vals):
            if len(x_vals) < 2:
                return []
            x_array = np.array(x_vals)
            y_array = np.array(y_vals)
            coefficients = np.polyfit(x_array, y_array, 1)
            x_min, x_max = x_array.min(), x_array.max()
            return [
                {'x': float(x_min), 'y': float(coefficients[0] * x_min + coefficients[1])},
                {'x': float(x_max), 'y': float(coefficients[0] * x_max + coefficients[1])}
            ]
        
        # Temperature trend line
        temp_x = [point['x'] for point in scatter_data['temperature']]
        temp_y = [point['y'] for point in scatter_data['temperature']]
        temp_trend = calculate_trend_line(temp_x, temp_y)
        
        # Wind speed trend line  
        wind_x = [point['x'] for point in scatter_data['wind_speed']]
        wind_y = [point['y'] for point in scatter_data['wind_speed']]
        wind_trend = calculate_trend_line(wind_x, wind_y) if wind_x else []
        
        # Temperature ranges analysis for additional stats
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
            'scatter_data': scatter_data,
            'trend_lines': {
                'temperature': temp_trend,
                'wind_speed': wind_trend
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

@app.route('/api/data-explorer')
def data_explorer():
    """Advanced data exploration endpoint with filtering and pagination"""
    try:
        db = get_db_connection()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Get query parameters for filtering
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        skip = (page - 1) * limit
        
        # Build MongoDB filter query from request parameters
        filter_query = {}
        
        # Building ID filters
        building_id_min = request.args.get('building_id_min')
        building_id_max = request.args.get('building_id_max')
        if building_id_min or building_id_max:
            building_filter = {}
            if building_id_min:
                building_filter['$gte'] = int(building_id_min)
            if building_id_max:
                building_filter['$lte'] = int(building_id_max)
            filter_query['building_id'] = building_filter
        
        # Primary use filter
        primary_use = request.args.get('primary_use')
        if primary_use:
            filter_query['primary_use'] = primary_use
        
        # Meter reading filters
        reading_min = request.args.get('meter_reading_min')
        reading_max = request.args.get('meter_reading_max')
        if reading_min or reading_max:
            reading_filter = {}
            if reading_min:
                reading_filter['$gte'] = float(reading_min)
            if reading_max:
                reading_filter['$lte'] = float(reading_max)
            filter_query['meter_reading'] = reading_filter
        
        # Weather filters
        temp_min = request.args.get('temp_min')
        temp_max = request.args.get('temp_max')
        if temp_min or temp_max:
            temp_filter = {}
            if temp_min:
                temp_filter['$gte'] = float(temp_min)
            if temp_max:
                temp_filter['$lte'] = float(temp_max)
            filter_query['air_temperature'] = temp_filter
        
        wind_min = request.args.get('wind_speed_min')
        wind_max = request.args.get('wind_speed_max')
        if wind_min or wind_max:
            wind_filter = {}
            if wind_min:
                wind_filter['$gte'] = float(wind_min)
            if wind_max:
                wind_filter['$lte'] = float(wind_max)
            filter_query['wind_speed'] = wind_filter
        
        # Date range filters
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter['$gte'] = datetime.strptime(date_from, '%Y-%m-%d')
            if date_to:
                date_filter['$lt'] = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            filter_query['timestamp'] = date_filter
        
        # Anomaly filter
        anomaly_filter = request.args.get('anomaly_filter')
        if anomaly_filter != '' and anomaly_filter is not None:
            filter_query['anomaly'] = int(anomaly_filter)
        
        print(f"🔍 Data explorer query: {filter_query}")
        
        # Get total count for pagination
        total_count = db.meter_readings.count_documents(filter_query)
        
        # Get filtered and paginated data
        cursor = db.meter_readings.find(filter_query).skip(skip).limit(limit).sort('timestamp', -1)
        records = list(cursor)
        
        # Calculate summary statistics for filtered data
        pipeline = [
            {'$match': filter_query},
            {'$group': {
                '_id': None,
                'total_records': {'$sum': 1},
                'unique_buildings': {'$addToSet': '$building_id'},
                'total_anomalies': {'$sum': {'$cond': [{'$eq': ['$anomaly', 1]}, 1, 0]}},
                'avg_consumption': {'$avg': '$meter_reading'},
                'min_consumption': {'$min': '$meter_reading'},
                'max_consumption': {'$max': '$meter_reading'}
            }}
        ]
        
        stats_result = list(db.meter_readings.aggregate(pipeline))
        if stats_result:
            stats = stats_result[0]
            stats['unique_buildings'] = len(stats['unique_buildings'])
        else:
            stats = {
                'total_records': 0,
                'unique_buildings': 0,
                'total_anomalies': 0,
                'avg_consumption': 0,
                'min_consumption': 0,
                'max_consumption': 0
            }
        
        # Format records for frontend
        formatted_records = []
        for record in records:
            formatted_records.append({
                'building_id': record.get('building_id'),
                'meter_reading': round(record.get('meter_reading', 0), 2),
                'timestamp': record.get('timestamp').isoformat() if record.get('timestamp') else None,
                'air_temperature': round(record.get('air_temperature', 0), 1) if record.get('air_temperature') else None,
                'wind_speed': round(record.get('wind_speed', 0), 1) if record.get('wind_speed') else None,
                'primary_use': record.get('primary_use', 'Unknown'),
                'anomaly': record.get('anomaly', 0)
            })
        
        return jsonify({
            'success': True,
            'data': formatted_records,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_count,
                'pages': (total_count + limit - 1) // limit
            },
            'statistics': {
                'total_records': stats['total_records'],
                'unique_buildings': stats['unique_buildings'],
                'total_anomalies': stats['total_anomalies'],
                'avg_consumption': round(stats['avg_consumption'], 1) if stats['avg_consumption'] else 0,
                'min_consumption': round(stats['min_consumption'], 2) if stats['min_consumption'] else 0,
                'max_consumption': round(stats['max_consumption'], 2) if stats['max_consumption'] else 0
            },
            'filters_applied': filter_query,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in data explorer: {e}")
        return jsonify({
            'error': 'Failed to fetch data',
            'message': str(e)
        }), 500

@app.route('/api/data-explorer-full')
def data_explorer_full():
    """Advanced data exploration endpoint with filtering and pagination"""
    try:
        db = get_db_connection()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Get query parameters for filtering
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        skip = (page - 1) * limit
        
        # Build MongoDB filter query from request parameters
        filter_query = {}
        
        # Building ID filters
        building_id_min = request.args.get('building_id_min')
        building_id_max = request.args.get('building_id_max')
        if building_id_min or building_id_max:
            building_filter = {}
            if building_id_min:
                building_filter['$gte'] = int(building_id_min)
            if building_id_max:
                building_filter['$lte'] = int(building_id_max)
            filter_query['building_id'] = building_filter
        
        # Primary use filter
        primary_use = request.args.get('primary_use')
        if primary_use:
            filter_query['primary_use'] = primary_use
        
        # Meter reading filters
        reading_min = request.args.get('meter_reading_min')
        reading_max = request.args.get('meter_reading_max')
        if reading_min or reading_max:
            reading_filter = {}
            if reading_min:
                reading_filter['$gte'] = float(reading_min)
            if reading_max:
                reading_filter['$lte'] = float(reading_max)
            filter_query['meter_reading'] = reading_filter
        
        # Weather filters
        temp_min = request.args.get('temp_min')
        temp_max = request.args.get('temp_max')
        if temp_min or temp_max:
            temp_filter = {}
            if temp_min:
                temp_filter['$gte'] = float(temp_min)
            if temp_max:
                temp_filter['$lte'] = float(temp_max)
            filter_query['air_temperature'] = temp_filter
        
        wind_min = request.args.get('wind_speed_min')
        wind_max = request.args.get('wind_speed_max')
        if wind_min or wind_max:
            wind_filter = {}
            if wind_min:
                wind_filter['$gte'] = float(wind_min)
            if wind_max:
                wind_filter['$lte'] = float(wind_max)
            filter_query['wind_speed'] = wind_filter
        
        # Date range filters
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter['$gte'] = datetime.strptime(date_from, '%Y-%m-%d')
            if date_to:
                date_filter['$lt'] = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            filter_query['timestamp'] = date_filter
        
        # Anomaly filter
        anomaly_filter = request.args.get('anomaly_filter')
        if anomaly_filter != '' and anomaly_filter is not None:
            filter_query['anomaly'] = int(anomaly_filter)
        
        print(f"🔍 Data explorer query: {filter_query}")
        
        # Get total count for pagination
        total_count = db.meter_readings.count_documents(filter_query)
        
        # Get filtered and paginated data
        cursor = db.meter_readings.find(filter_query).skip(skip).limit(limit).sort('timestamp', -1)
        records = list(cursor)
        
        # Calculate summary statistics for filtered data
        pipeline = [
            {'$match': filter_query},
            {'$group': {
                '_id': None,
                'total_records': {'$sum': 1},
                'unique_buildings': {'$addToSet': '$building_id'},
                'total_anomalies': {'$sum': {'$cond': [{'$eq': ['$anomaly', 1]}, 1, 0]}},
                'avg_consumption': {'$avg': '$meter_reading'},
                'min_consumption': {'$min': '$meter_reading'},
                'max_consumption': {'$max': '$meter_reading'}
            }}
        ]
        
        stats_result = list(db.meter_readings.aggregate(pipeline))
        if stats_result:
            stats = stats_result[0]
            stats['unique_buildings'] = len(stats['unique_buildings'])
        else:
            stats = {
                'total_records': 0,
                'unique_buildings': 0,
                'total_anomalies': 0,
                'avg_consumption': 0,
                'min_consumption': 0,
                'max_consumption': 0
            }
        
        # Format records for frontend
        formatted_records = []
        for record in records:
            formatted_records.append({
                'building_id': record.get('building_id'),
                'meter_reading': round(record.get('meter_reading', 0), 2),
                'timestamp': record.get('timestamp').isoformat() if record.get('timestamp') else None,
                'air_temperature': round(record.get('air_temperature', 0), 1) if record.get('air_temperature') else None,
                'wind_speed': round(record.get('wind_speed', 0), 1) if record.get('wind_speed') else None,
                'primary_use': record.get('primary_use', 'Unknown'),
                'anomaly': record.get('anomaly', 0)
            })
        
        return jsonify({
            'success': True,
            'data': formatted_records,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_count,
                'pages': (total_count + limit - 1) // limit
            },
            'statistics': {
                'total_records': stats['total_records'],
                'unique_buildings': stats['unique_buildings'],
                'total_anomalies': stats['total_anomalies'],
                'avg_consumption': round(stats['avg_consumption'], 1) if stats['avg_consumption'] else 0,
                'min_consumption': round(stats['min_consumption'], 2) if stats['min_consumption'] else 0,
                'max_consumption': round(stats['max_consumption'], 2) if stats['max_consumption'] else 0
            },
            'filters_applied': filter_query,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in data explorer: {e}")
        return jsonify({
            'error': 'Failed to fetch data',
            'message': str(e)
        }), 500
    """Advanced data exploration endpoint with filtering and pagination"""
    try:
        db = get_db_connection()
        if db is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        # Get query parameters for filtering
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        skip = (page - 1) * limit
        
        # Build MongoDB filter query from request parameters
        filter_query = {}
        
        # Building ID filters
        building_id_min = request.args.get('building_id_min')
        building_id_max = request.args.get('building_id_max')
        if building_id_min or building_id_max:
            building_filter = {}
            if building_id_min:
                building_filter['$gte'] = int(building_id_min)
            if building_id_max:
                building_filter['$lte'] = int(building_id_max)
            filter_query['building_id'] = building_filter
        
        # Primary use filter
        primary_use = request.args.get('primary_use')
        if primary_use:
            filter_query['primary_use'] = primary_use
        
        # Meter reading filters
        reading_min = request.args.get('meter_reading_min')
        reading_max = request.args.get('meter_reading_max')
        if reading_min or reading_max:
            reading_filter = {}
            if reading_min:
                reading_filter['$gte'] = float(reading_min)
            if reading_max:
                reading_filter['$lte'] = float(reading_max)
            filter_query['meter_reading'] = reading_filter
        
        # Weather filters
        temp_min = request.args.get('temp_min')
        temp_max = request.args.get('temp_max')
        if temp_min or temp_max:
            temp_filter = {}
            if temp_min:
                temp_filter['$gte'] = float(temp_min)
            if temp_max:
                temp_filter['$lte'] = float(temp_max)
            filter_query['air_temperature'] = temp_filter
        
        wind_min = request.args.get('wind_speed_min')
        wind_max = request.args.get('wind_speed_max')
        if wind_min or wind_max:
            wind_filter = {}
            if wind_min:
                wind_filter['$gte'] = float(wind_min)
            if wind_max:
                wind_filter['$lte'] = float(wind_max)
            filter_query['wind_speed'] = wind_filter
        
        # Date range filters
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter['$gte'] = datetime.strptime(date_from, '%Y-%m-%d')
            if date_to:
                date_filter['$lt'] = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            filter_query['timestamp'] = date_filter
        
        # Anomaly filter
        anomaly_filter = request.args.get('anomaly_filter')
        if anomaly_filter != '' and anomaly_filter is not None:
            filter_query['anomaly'] = int(anomaly_filter)
        
        print(f"🔍 Data explorer query: {filter_query}")
        
        # Get total count for pagination
        total_count = db.meter_readings.count_documents(filter_query)
        
        # Get filtered and paginated data
        cursor = db.meter_readings.find(filter_query).skip(skip).limit(limit).sort('timestamp', -1)
        records = list(cursor)
        
        # Calculate summary statistics for filtered data
        pipeline = [
            {'$match': filter_query},
            {'$group': {
                '_id': None,
                'total_records': {'$sum': 1},
                'unique_buildings': {'$addToSet': '$building_id'},
                'total_anomalies': {'$sum': {'$cond': [{'$eq': ['$anomaly', 1]}, 1, 0]}},
                'avg_consumption': {'$avg': '$meter_reading'},
                'min_consumption': {'$min': '$meter_reading'},
                'max_consumption': {'$max': '$meter_reading'}
            }}
        ]
        
        stats_result = list(db.meter_readings.aggregate(pipeline))
        if stats_result:
            stats = stats_result[0]
            stats['unique_buildings'] = len(stats['unique_buildings'])
        else:
            stats = {
                'total_records': 0,
                'unique_buildings': 0,
                'total_anomalies': 0,
                'avg_consumption': 0,
                'min_consumption': 0,
                'max_consumption': 0
            }
        
        # Format records for frontend
        formatted_records = []
        for record in records:
            formatted_records.append({
                'building_id': record.get('building_id'),
                'meter_reading': round(record.get('meter_reading', 0), 2),
                'timestamp': record.get('timestamp').isoformat() if record.get('timestamp') else None,
                'air_temperature': round(record.get('air_temperature', 0), 1) if record.get('air_temperature') else None,
                'wind_speed': round(record.get('wind_speed', 0), 1) if record.get('wind_speed') else None,
                'primary_use': record.get('primary_use', 'Unknown'),
                'anomaly': record.get('anomaly', 0)
            })
        
        return jsonify({
            'success': True,
            'data': formatted_records,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total_count,
                'pages': (total_count + limit - 1) // limit
            },
            'statistics': {
                'total_records': stats['total_records'],
                'unique_buildings': stats['unique_buildings'],
                'total_anomalies': stats['total_anomalies'],
                'avg_consumption': round(stats['avg_consumption'], 1) if stats['avg_consumption'] else 0,
                'min_consumption': round(stats['min_consumption'], 2) if stats['min_consumption'] else 0,
                'max_consumption': round(stats['max_consumption'], 2) if stats['max_consumption'] else 0
            },
            'filters_applied': filter_query,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in data explorer: {e}")
        return jsonify({
            'error': 'Failed to fetch data',
            'message': str(e)
        }), 500


# ==================== ML PREDICTION ENDPOINTS ====================

@app.route('/api/ml/status', methods=['GET'])
def ml_status():
    """Get ML model status and metrics"""
    if ml_api is None:
        return jsonify({
            'error': 'ML API not available',
            'message': 'XGBoost dependencies not installed'
        }), 503
    
    try:
        status = ml_api.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'error': 'Failed to get ML status',
            'message': str(e)
        }), 500


@app.route('/api/ml/train', methods=['POST'])
def ml_train():
    """Train the XGBoost model"""
    if ml_api is None:
        return jsonify({
            'error': 'ML API not available',
            'message': 'XGBoost dependencies not installed'
        }), 503
    
    try:
        data = request.get_json()
        sample_size = data.get('sample_size') if data else None
        
        # Convert sample_size to int if provided
        if sample_size:
            sample_size = int(sample_size)
        
        print(f"🚀 Training ML model with sample_size: {sample_size}")
        
        # Train model
        metrics = ml_api.train_model(sample_size=sample_size)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': metrics
        })
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return jsonify({
            'error': 'Failed to train model',
            'message': str(e)
        }), 500


@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    """Make predictions using the trained model"""
    if ml_api is None:
        return jsonify({
            'error': 'ML API not available',
            'message': 'XGBoost dependencies not installed'
        }), 503
    
    try:
        data = request.get_json() or {}
        
        prediction_type = data.get('type', 'recent')
        building_id = data.get('building_id')
        limit = int(data.get('limit', 100))
        date_from = data.get('date_from')
        date_to = data.get('date_to')
        
        print(f"🔮 Making predictions: type={prediction_type}, limit={limit}")
        
        # Make predictions
        results = ml_api.predict(
            prediction_type=prediction_type,
            building_id=building_id,
            limit=limit,
            date_from=date_from,
            date_to=date_to
        )
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return jsonify({
            'error': 'Failed to make predictions',
            'message': str(e)
        }), 500


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
        print("   http://localhost:5000/api/weather-analysis")
        print("   http://localhost:5000/api/stream-mining-demo")
        print("   http://localhost:5000/api/performance-stats")
        print("   http://localhost:5000/api/meter-readings [POST]")
        print("   http://localhost:5000/api/data-explorer")
        print("   🤖 ML Endpoints:")
        print("   http://localhost:5000/api/ml/status")
        print("   http://localhost:5000/api/ml/train [POST]")
        print("   http://localhost:5000/api/ml/predict [POST]")
        
        # Disable debug mode to prevent restart issues
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start API - MongoDB connection failed")
        print("💡 Make sure MongoDB is running on localhost:27017")