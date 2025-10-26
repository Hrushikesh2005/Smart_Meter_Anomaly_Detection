"""
Enhanced Analytics for Smart Meter Data
Weather correlation and building performance analysis
"""

import pandas as pd
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SmartMeterAnalytics:
    """
    Advanced analytics for smart meter anomaly detection system
    """
    
    def __init__(self, connection_string="mongodb://localhost:27017/"):
        self.client = MongoClient(connection_string)
        self.db = self.client['smart_meter_db']
        
    def load_sample_data(self, limit=10000):
        """Load sample data for analysis"""
        
        print("📊 Loading sample data for analysis...")
        
        # Load data with weather and building features
        pipeline = [
            {'$match': {'meter_reading': {'$ne': None}}},
            {'$sample': {'size': limit}},
            {
                '$project': {
                    'building_id': 1,
                    'meter_reading': 1,
                    'anomaly': 1,
                    'air_temperature': 1,
                    'wind_speed': 1,
                    'cloud_coverage': 1,
                    'primary_use': 1,
                    'square_feet': 1,
                    'floor_count': 1,
                    'hour': 1,
                    'weekday': 1,
                    'month': 1,
                    'timestamp': 1
                }
            }
        ]
        
        data = list(self.db['meter_readings'].aggregate(pipeline))
        df = pd.DataFrame(data)
        
        print(f"✅ Loaded {len(df)} records for analysis")
        return df
    
    def weather_correlation_analysis(self, df):
        """Analyze correlation between weather and energy consumption"""
        
        print("\n🌤️ Weather Correlation Analysis")
        print("=" * 40)
        
        # Filter out null values
        weather_df = df.dropna(subset=['meter_reading', 'air_temperature', 'wind_speed'])
        
        if len(weather_df) < 100:
            print("❌ Insufficient data for weather analysis")
            return {}
        
        # Calculate correlations
        correlations = {}
        
        # Temperature correlation
        temp_corr = weather_df['meter_reading'].corr(weather_df['air_temperature'])
        correlations['temperature'] = temp_corr
        print(f"🌡️  Temperature vs Energy: {temp_corr:.3f}")
        
        # Wind speed correlation
        wind_corr = weather_df['meter_reading'].corr(weather_df['wind_speed'])
        correlations['wind_speed'] = wind_corr
        print(f"💨 Wind Speed vs Energy: {wind_corr:.3f}")
        
        # Cloud coverage correlation
        if 'cloud_coverage' in weather_df.columns:
            cloud_corr = weather_df['meter_reading'].corr(weather_df['cloud_coverage'])
            correlations['cloud_coverage'] = cloud_corr
            print(f"☁️  Cloud Coverage vs Energy: {cloud_corr:.3f}")
        
        # Temperature ranges analysis
        weather_df['temp_range'] = pd.cut(weather_df['air_temperature'], 
                                         bins=5, labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'])
        
        temp_consumption = weather_df.groupby('temp_range')['meter_reading'].agg(['mean', 'std', 'count'])
        print(f"\n🌡️  Energy Consumption by Temperature Range:")
        for temp_range, row in temp_consumption.iterrows():
            if pd.notna(row['mean']):
                print(f"   {temp_range}: {row['mean']:.1f} ± {row['std']:.1f} kWh ({row['count']} readings)")
        
        return {
            'correlations': correlations,
            'temperature_analysis': temp_consumption.to_dict(),
            'sample_size': len(weather_df)
        }
    
    def building_performance_analysis(self, df):
        """Analyze building performance across different types"""
        
        print("\n🏢 Building Performance Analysis")
        print("=" * 40)
        
        # Performance by building type
        building_stats = df.groupby('primary_use').agg({
            'meter_reading': ['mean', 'std', 'count'],
            'anomaly': 'sum',
            'square_feet': 'mean'
        }).round(2)
        
        # Flatten column names
        building_stats.columns = ['_'.join(col).strip() for col in building_stats.columns.values]
        
        # Calculate metrics
        building_stats['anomaly_rate'] = (building_stats['anomaly_sum'] / 
                                        building_stats['meter_reading_count'] * 100).round(2)
        
        building_stats['consumption_per_sqft'] = (building_stats['meter_reading_mean'] / 
                                                building_stats['square_feet_mean'] * 1000).round(3)
        
        print("📊 Performance by Building Type:")
        print(f"{'Type':<25} {'Avg kWh':<10} {'Anomaly %':<10} {'kWh/sqft':<10} {'Count':<8}")
        print("-" * 70)
        
        for building_type, row in building_stats.iterrows():
            if pd.notna(row['meter_reading_mean']):
                print(f"{building_type:<25} {row['meter_reading_mean']:<10.1f} "
                      f"{row['anomaly_rate']:<10.1f} {row['consumption_per_sqft']:<10.3f} "
                      f"{row['meter_reading_count']:<8.0f}")
        
        # Building size analysis
        df['size_category'] = pd.cut(df['square_feet'], 
                                   bins=[0, 10000, 50000, 100000, float('inf')], 
                                   labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        size_stats = df.groupby('size_category').agg({
            'meter_reading': ['mean', 'count'],
            'anomaly': 'sum'
        })
        
        print(f"\n📏 Performance by Building Size:")
        for size, row in size_stats.iterrows():
            if pd.notna(row[('meter_reading', 'mean')]):
                count = row[('meter_reading', 'count')]
                anomalies = row[('anomaly', 'sum')]
                avg_consumption = row[('meter_reading', 'mean')]
                anomaly_rate = (anomalies / count * 100) if count > 0 else 0
                
                print(f"   {size}: {avg_consumption:.1f} kWh avg, "
                      f"{anomaly_rate:.1f}% anomaly rate ({count} buildings)")
        
        return {
            'building_type_stats': building_stats.to_dict(),
            'size_category_stats': size_stats.to_dict()
        }
    
    def temporal_pattern_analysis(self, df):
        """Analyze temporal patterns in energy consumption"""
        
        print("\n⏰ Temporal Pattern Analysis")
        print("=" * 40)
        
        # Hourly patterns
        hourly_stats = df.groupby('hour').agg({
            'meter_reading': ['mean', 'count'],
            'anomaly': 'sum'
        })
        
        print("🕐 Hourly Consumption Patterns:")
        peak_hours = []
        
        for hour in range(24):
            if hour in hourly_stats.index:
                row = hourly_stats.loc[hour]
                avg_consumption = row[('meter_reading', 'mean')]
                count = row[('meter_reading', 'count')]
                
                if count > 10 and avg_consumption > df['meter_reading'].mean():
                    peak_hours.append(hour)
        
        if peak_hours:
            print(f"   Peak hours: {', '.join(map(str, peak_hours))}")
        else:
            print("   No clear peak hours identified")
        
        # Weekly patterns
        weekday_stats = df.groupby('weekday').agg({
            'meter_reading': ['mean', 'count'],
            'anomaly': 'sum'
        })
        
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        print(f"\n📅 Weekly Consumption Patterns:")
        
        for weekday in range(7):
            if weekday in weekday_stats.index:
                row = weekday_stats.loc[weekday]
                avg_consumption = row[('meter_reading', 'mean')]
                count = row[('meter_reading', 'count')]
                
                if count > 10:
                    day_name = weekday_names[weekday] if weekday < 7 else f"Day {weekday}"
                    print(f"   {day_name}: {avg_consumption:.1f} kWh avg ({count} readings)")
        
        return {
            'hourly_patterns': hourly_stats.to_dict(),
            'weekly_patterns': weekday_stats.to_dict(),
            'peak_hours': peak_hours
        }
    
    def anomaly_analysis(self, df):
        """Detailed analysis of anomalies"""
        
        print("\n🚨 Anomaly Analysis")
        print("=" * 40)
        
        total_readings = len(df)
        total_anomalies = df['anomaly'].sum()
        anomaly_rate = (total_anomalies / total_readings * 100) if total_readings > 0 else 0
        
        print(f"📊 Overall Statistics:")
        print(f"   Total readings analyzed: {total_readings:,}")
        print(f"   Total anomalies: {total_anomalies:,}")
        print(f"   Anomaly rate: {anomaly_rate:.2f}%")
        
        if total_anomalies == 0:
            print("   No anomalies found in sample data")
            return {'anomaly_rate': anomaly_rate}
        
        # Anomaly characteristics
        anomalies_df = df[df['anomaly'] == 1]
        normal_df = df[df['anomaly'] == 0]
        
        if len(anomalies_df) > 0 and len(normal_df) > 0:
            print(f"\n🔍 Anomaly Characteristics:")
            
            # Consumption comparison
            anomaly_avg = anomalies_df['meter_reading'].mean()
            normal_avg = normal_df['meter_reading'].mean()
            
            print(f"   Anomaly avg consumption: {anomaly_avg:.1f} kWh")
            print(f"   Normal avg consumption: {normal_avg:.1f} kWh")
            print(f"   Difference: {anomaly_avg - normal_avg:.1f} kWh ({((anomaly_avg/normal_avg - 1) * 100):.1f}%)")
            
            # Building type distribution
            anomaly_buildings = anomalies_df['primary_use'].value_counts()
            print(f"\n🏢 Anomalies by Building Type:")
            for building_type, count in anomaly_buildings.head().items():
                total_type = df[df['primary_use'] == building_type].shape[0]
                rate = (count / total_type * 100) if total_type > 0 else 0
                print(f"   {building_type}: {count} anomalies ({rate:.1f}% of type)")
        
        return {
            'anomaly_rate': anomaly_rate,
            'total_anomalies': total_anomalies,
            'anomaly_avg_consumption': anomaly_avg if 'anomaly_avg' in locals() else None,
            'normal_avg_consumption': normal_avg if 'normal_avg' in locals() else None
        }
    
    def generate_insights(self, df):
        """Generate actionable insights from the analysis"""
        
        print("\n💡 Actionable Insights")
        print("=" * 40)
        
        insights = []
        
        # Energy efficiency insights
        building_efficiency = df.groupby('primary_use')['meter_reading'].mean()
        if len(building_efficiency) > 1:
            most_efficient = building_efficiency.idxmin()
            least_efficient = building_efficiency.idxmax()
            
            insights.append(f"🏆 Most efficient building type: {most_efficient}")
            insights.append(f"⚠️  Least efficient building type: {least_efficient}")
        
        # Anomaly insights
        if df['anomaly'].sum() > 0:
            high_anomaly_buildings = df.groupby('building_id')['anomaly'].sum().sort_values(ascending=False)
            if len(high_anomaly_buildings) > 0:
                worst_building = high_anomaly_buildings.index[0]
                anomaly_count = high_anomaly_buildings.iloc[0]
                insights.append(f"🚨 Building {worst_building} has {anomaly_count} anomalies - needs attention")
        
        # Weather insights
        if 'air_temperature' in df.columns:
            temp_corr = df['meter_reading'].corr(df['air_temperature'])
            if abs(temp_corr) > 0.3:
                direction = "increases" if temp_corr > 0 else "decreases"
                insights.append(f"🌡️  Energy consumption {direction} with temperature (correlation: {temp_corr:.2f})")
        
        # Time-based insights
        if 'hour' in df.columns:
            hourly_avg = df.groupby('hour')['meter_reading'].mean()
            if len(hourly_avg) > 12:
                peak_hour = hourly_avg.idxmax()
                insights.append(f"⏰ Peak consumption hour: {peak_hour}:00")
        
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        if not insights:
            print("   No specific insights generated from current data sample")
        
        return insights
    
    def run_complete_analysis(self, sample_size=5000):
        """Run complete analytics pipeline"""
        
        print("🔬 Smart Meter Advanced Analytics")
        print("=" * 60)
        print(f"Analyzing sample of {sample_size:,} records")
        print("=" * 60)
        
        # Load data
        df = self.load_sample_data(sample_size)
        
        if len(df) < 100:
            print("❌ Insufficient data for analysis")
            return {}
        
        # Run all analyses
        weather_results = self.weather_correlation_analysis(df)
        building_results = self.building_performance_analysis(df)
        temporal_results = self.temporal_pattern_analysis(df)
        anomaly_results = self.anomaly_analysis(df)
        insights = self.generate_insights(df)
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"✅ Weather correlation analysis completed")
        print(f"✅ Building performance analysis completed")
        print(f"✅ Temporal pattern analysis completed")
        print(f"✅ Anomaly analysis completed")
        print(f"✅ {len(insights)} actionable insights generated")
        
        return {
            'weather': weather_results,
            'buildings': building_results,
            'temporal': temporal_results,
            'anomalies': anomaly_results,
            'insights': insights,
            'sample_size': len(df)
        }

def run_analytics_demo():
    """Run the analytics demonstration"""
    
    print("📊 Starting Enhanced Analytics Demo...")
    
    try:
        analytics = SmartMeterAnalytics()
        results = analytics.run_complete_analysis(sample_size=3000)
        
        print("\n🎉 Analytics demo completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Analytics demo failed: {e}")
        return {}

if __name__ == "__main__":
    run_analytics_demo()