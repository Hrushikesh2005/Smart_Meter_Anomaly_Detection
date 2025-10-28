"""
System Check Script - Verify Smart Meter Anomaly Detection Setup
Run this to ensure everything is configured correctly
"""

import sys
import os
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_version():
    """Check if Python version is compatible"""
    print_section("1. Python Version Check")
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print_section("2. Dependencies Check")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'pymongo': 'pymongo',
        'flask': 'flask',
        'flask_cors': 'flask-cors',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            installed.append(package_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"❌ {package_name} - NOT INSTALLED")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing)}")
        return False
    else:
        print(f"\n✅ All {len(installed)} required packages are installed!")
        return True

def check_mongodb_connection():
    """Check if MongoDB is accessible"""
    print_section("3. MongoDB Connection Check")
    
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError
        
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=3000)
        client.server_info()
        
        # Check if database exists
        db = client['smart_meter_db']
        collections = db.list_collection_names()
        
        print("✅ MongoDB is running and accessible")
        print(f"📊 Database: smart_meter_db")
        
        if collections:
            print(f"📁 Collections found: {', '.join(collections)}")
            
            # Check collection sizes
            for coll_name in collections:
                count = db[coll_name].count_documents({})
                print(f"   - {coll_name}: {count:,} documents")
        else:
            print("⚠️  No collections found. Run database setup script:")
            print("   python updated_mongodb_setup.py")
        
        return True
        
    except ImportError:
        print("❌ pymongo is not installed")
        print("💡 Install it with: pip install pymongo")
        return False
    except Exception as e:
        error_msg = str(e)
        if 'ServerSelectionTimeoutError' in type(e).__name__:
            print("❌ MongoDB is not running or not accessible")
            print("💡 Make sure MongoDB is installed and running on localhost:27017")
        else:
            print(f"❌ MongoDB connection failed: {error_msg}")
        return False

def check_data_files():
    """Check if dataset files exist"""
    print_section("4. Dataset Files Check")
    
    data_dir = 'data'
    required_files = [
        'train.csv',
        'test.csv',
        'train_features.csv',
        'test_features.csv'
    ]
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    missing = []
    found = []
    
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            found.append(filename)
            print(f"✅ {filename} ({size_mb:.1f} MB)")
        else:
            missing.append(filename)
            print(f"❌ {filename} - NOT FOUND")
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        print("💡 Download dataset from Kaggle and place in data/ directory")
        return False
    else:
        print(f"\n✅ All {len(found)} dataset files found!")
        return True

def check_project_structure():
    """Check if all necessary project files exist"""
    print_section("5. Project Structure Check")
    
    required_structure = {
        'scripts': ['flask_api_setup.py', 'anomaly_detection.py', 'stream_mining.py', 'enhanced_analytics.py'],
        'frontend': ['dashboard.html'],
        'root': ['requirements.txt', 'README.md', 'updated_mongodb_setup.py']
    }
    
    all_good = True
    
    for location, files in required_structure.items():
        if location == 'root':
            base_path = '.'
        else:
            base_path = location
        
        for filename in files:
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                print(f"✅ {filepath}")
            else:
                print(f"❌ {filepath} - NOT FOUND")
                all_good = False
    
    return all_good

def test_flask_api():
    """Test if Flask API can be imported without errors"""
    print_section("6. Flask API Import Check")
    
    try:
        sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
        
        # Try importing key modules
        from flask_api_setup import app
        print("✅ Flask API imports successfully")
        
        from anomaly_detection import SimpleAnomalyDetector
        print("✅ Anomaly detection module imports successfully")
        
        from stream_mining import BloomFilter, DGIM
        print("✅ Stream mining module imports successfully")
        
        from enhanced_analytics import SmartMeterAnalytics
        print("✅ Analytics module imports successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def generate_summary(results):
    """Generate final summary and recommendations"""
    print_section("📋 SETUP SUMMARY")
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print(f"\nChecks Passed: {passed_checks}/{total_checks}")
    print(f"Status: {'✅ READY' if passed_checks == total_checks else '⚠️  NEEDS ATTENTION'}")
    
    if passed_checks == total_checks:
        print("\n🎉 Your system is ready to run!")
        print("\n📋 Next Steps:")
        print("   1. Start MongoDB (if not already running)")
        print("   2. Load data: python updated_mongodb_setup.py")
        print("   3. Start API: python scripts/flask_api_setup.py")
        print("   4. Open dashboard: frontend/dashboard.html")
    else:
        print("\n⚠️  Please fix the issues above before proceeding")
        
        # Specific recommendations
        if not results['dependencies']:
            print("\n💡 Install dependencies:")
            print("   pip install -r requirements.txt")
        
        if not results['mongodb']:
            print("\n💡 MongoDB not running:")
            print("   - Windows: Start MongoDB service")
            print("   - Or download from: https://www.mongodb.com/try/download/community")
        
        if not results['data_files']:
            print("\n💡 Download dataset:")
            print("   - From: Kaggle ASHRAE competition or LEAD dataset")
            print("   - Place in: data/ directory")

def main():
    """Main check routine"""
    print("\n" + "="*60)
    print("  🔍 Smart Meter Anomaly Detection - System Check")
    print("="*60)
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Run all checks
    results = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'mongodb': check_mongodb_connection(),
        'data_files': check_data_files(),
        'structure': check_project_structure(),
        'imports': test_flask_api()
    }
    
    # Generate summary
    generate_summary(results)
    
    print("\n" + "="*60)
    print("  Check Complete!")
    print("="*60 + "\n")
    
    return all(results.values())

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during check: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
