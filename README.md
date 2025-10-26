# 🎯 Smart Meter Anomaly Detection System

## 🌟 **Complete IoT Anomaly Detection Platform with Advanced Analytics**

A production-ready smart meter anomaly detection system featuring:
- **Real-time MongoDB integration** with 1.7M+ meter readings
- **Advanced weather correlation analysis**
- **Big Data stream mining algorithms** (Bloom Filter + DGIM)
- **Interactive dashboard** with live charts and monitoring
- **RESTful API** with optimized performance and caching

---

## 🚀 **Quick Start Guide**

### **Step 1: Prerequisites Setup**

```bash
# Install Python dependencies
pip install -r requirements.txt

# Required packages:
# - pymongo (MongoDB integration)
# - flask, flask-cors (API server)
# - pandas, numpy (data processing)
# - scikit-learn (machine learning)
```

### **Step 2: Database Setup**

```bash
# Load dataset and setup MongoDB (1.7M records)
python scripts/updated_mongodb_setup.py
```

**Expected Output:**
```
✅ Connected to MongoDB
✅ Created database: smart_meter_db
✅ Loaded 1,749,494 meter readings
✅ 200 buildings with weather data
✅ Database setup complete
```

### **Step 3: Start the System**

```bash
# Activate virtual environment (if using)
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Start Flask API server
python scripts/flask_api_setup.py
```

**Expected Output:**
```
✅ MongoDB connected successfully
✅ Loaded 200 buildings from database
🚀 API server running on http://localhost:5000

Available endpoints:
- GET /api/stats
- GET /api/buildings  
- GET /api/weather-analysis
- GET /api/stream-mining-demo
- GET /api/performance-stats
```

### **Step 4: Open Dashboard**

```bash
# Open the dashboard in your browser
start frontend/dashboard.html
# Or manually navigate to: file:///path/to/frontend/dashboard.html
```

---

## 🌐 **API Endpoints Reference**

| Endpoint | Method | Description | Response Format |
|----------|--------|-------------|----------------|
| `/api/stats` | GET | System overview and statistics | `{"total_buildings": 200, "total_anomalies": 8547, ...}` |
| `/api/buildings` | GET | Building list with status | `[{"building_id": 1, "status": "normal", ...}, ...]` |
| `/api/weather-analysis` | GET | **Live weather correlation analysis** | `{"correlations": {"temperature": -0.045, ...}, ...}` |
| `/api/stream-mining-demo` | GET | **Stream mining algorithms demo** | `{"bloom_filter_stats": {...}, "dgim_stats": {...}}` |
| `/api/performance-stats` | GET | Performance metrics and optimization | `{"response_times": {...}, "cache_stats": {...}}` |

### **Test API Endpoints**

```bash
# Test core system
curl http://localhost:5000/api/stats

# Test weather analysis (Phase 3)
curl http://localhost:5000/api/weather-analysis

# Test stream mining (Phase 3)  
curl http://localhost:5000/api/stream-mining-demo
```

---

## 📊 **Dashboard Features**

### **Core Monitoring**
- **Building Distribution Chart**: Real-time building type distribution
- **Anomaly Trends**: Time-series anomaly detection over time
- **Live Statistics**: Active buildings (200), total anomalies (8,547), recent alerts

### **Advanced Analytics (Phase 3)**
- **Weather Correlation Chart**: Live temperature and wind speed correlations
- **Stream Mining Statistics**: 
  - Bloom Filter performance (duplicate detection)
  - DGIM Algorithm metrics (sliding window counting)
- **Auto-refresh**: Configurable rates (5s, 10s, 30s, 60s)

---

## 🏗️ **System Architecture**

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Frontend          │    │   Flask API         │    │   MongoDB           │
│   dashboard.html    │◄──►│   flask_api_setup   │◄──►│   smart_meter_db    │
│   - Chart.js        │    │   - Core endpoints  │    │   - 1.7M records    │
│   - Real-time UI    │    │   - Weather API     │    │   - 200 buildings   │
│   - Phase 3 features│    │   - Stream mining   │    │   - Weather data    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

---

## � **Advanced Usage**

### **Run Individual Components**

```bash
# Enhanced analytics standalone
python scripts/enhanced_analytics.py

# Stream mining algorithms demo
python scripts/stream_mining.py

# Explore dataset structure
python scripts/explore_lead_dataset.py
```

### **Development Mode**

```bash
# API with debug mode
python scripts/flask_api_setup.py --debug

# Custom refresh rate on dashboard
# Edit dashboard.html, change: refreshRate = 5; // seconds
```

---

## 📈 **Data Overview**

### **Database Statistics**
- **Total Meter Readings**: 1,749,494 records
- **Buildings Monitored**: 200 unique buildings
- **Time Range**: Complete year of hourly readings
- **Weather Data**: Temperature, wind speed, humidity, pressure
- **Anomaly Rate**: ~0.49% (8,547 anomalies detected)

### **Building Types Distribution**
- Office buildings, retail spaces, educational facilities
- Mixed-use commercial buildings
- Various sizes and energy usage patterns

---

## 🎯 **Phase Implementation Status**

### ✅ **Phase 1: Core Infrastructure** 
- MongoDB setup with 1.7M meter readings
- Basic anomaly detection algorithms
- Core Flask API endpoints

### ✅ **Phase 2: Dashboard & Analytics**
- HTML Dashboard with Chart.js visualization  
- Stream mining algorithms (Bloom Filter + DGIM)
- Enhanced analytics capabilities

### ✅ **Phase 3: Advanced Integration**
- **Weather correlation analysis** - Live API integration
- **Stream mining demos** - Real-time algorithm statistics  
- **Performance optimization** - Caching and database indexing

---

## 🛠️ **Troubleshooting**

### **Common Issues**

**MongoDB Connection Error:**
```bash
# Ensure MongoDB is running
# Check connection string in scripts/flask_api_setup.py
```

**API Not Accessible:**
```bash
# Verify Flask server is running on http://localhost:5000
# Check virtual environment activation
# Ensure all dependencies installed
```

**Dashboard Not Loading Data:**
```bash
# Verify API endpoints respond with 200 OK
curl http://localhost:5000/api/stats
# Check browser console for JavaScript errors
```

**Missing Dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt
```

---

## 📚 **Technical Features**

### **Big Data Concepts**
- **Volume**: 1.7M+ records processed efficiently
- **Velocity**: Real-time streaming simulation with Bloom Filter
- **Variety**: Time-series, weather, and building metadata

### **Stream Mining Algorithms**
- **Bloom Filter**: Space-efficient duplicate detection
- **DGIM Algorithm**: Sliding window counting for big data streams
- **Memory Optimization**: Efficient bucket management

### **Weather Analytics**
- **Correlation Analysis**: Temperature vs energy consumption
- **Temporal Patterns**: Seasonal and daily usage variations
- **Building Performance**: Cross-building comparison analytics

---

## 🎉 **Success Indicators**

When everything is working correctly, you should see:

✅ **API Server**: Running on http://localhost:5000 with all 5 endpoints active  
✅ **Database**: 1,749,494 records loaded across 200 buildings  
✅ **Dashboard**: Live charts updating with real-time data  
✅ **Weather Analysis**: Temperature correlation showing ~-0.045  
✅ **Stream Mining**: Bloom Filter and DGIM statistics displaying  

**The system is now ready for production use!** 🚀

---

## 📞 **Support**

For issues or questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure MongoDB is accessible and populated
4. Check API endpoint responses with curl

**System demonstrates enterprise-grade IoT anomaly detection with advanced analytics capabilities.**