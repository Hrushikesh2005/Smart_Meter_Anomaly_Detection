"""
Stream Mining Algorithms for Smart Meter Anomaly Detection
Implements Bloom Filter and DGIM Algorithm for Big Data concepts
"""

import hashlib
import math
import time
from collections import deque
from datetime import datetime, timedelta
import numpy as np

class BloomFilter:
    """
    Bloom Filter for efficient duplicate anomaly detection
    Space-efficient probabilistic data structure
    """
    
    def __init__(self, capacity=10000, error_rate=0.1):
        """
        Initialize Bloom Filter
        
        Args:
            capacity: Expected number of elements
            error_rate: Desired false positive probability
        """
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Calculate optimal bit array size and hash functions
        self.bit_size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.hash_count = int(self.bit_size * math.log(2) / capacity)
        
        # Initialize bit array
        self.bit_array = [0] * self.bit_size
        self.count = 0
        
        print(f"🔍 Bloom Filter initialized:")
        print(f"   Capacity: {capacity:,}")
        print(f"   Error rate: {error_rate}")
        print(f"   Bit array size: {self.bit_size:,}")
        print(f"   Hash functions: {self.hash_count}")
    
    def _hash(self, item, seed):
        """Generate hash for item with seed"""
        hash_input = f"{item}_{seed}".encode('utf-8')
        return int(hashlib.md5(hash_input).hexdigest(), 16) % self.bit_size
    
    def add(self, item):
        """Add item to Bloom Filter"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = 1
        self.count += 1
    
    def contains(self, item):
        """Check if item might be in the set (can have false positives)"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True
    
    def estimated_false_positive_rate(self):
        """Calculate current false positive rate"""
        if self.count == 0:
            return 0.0
        
        # Probability that a bit is still 0
        prob_zero = (1 - 1/self.bit_size) ** (self.hash_count * self.count)
        
        # Probability of false positive
        return (1 - prob_zero) ** self.hash_count
    
    def get_stats(self):
        """Get Bloom Filter statistics"""
        bits_set = sum(self.bit_array)
        return {
            'capacity': self.capacity,
            'items_added': self.count,
            'bit_array_size': self.bit_size,
            'bits_set': bits_set,
            'utilization': bits_set / self.bit_size,
            'estimated_fpr': self.estimated_false_positive_rate(),
            'hash_functions': self.hash_count
        }

class DGIMBucket:
    """Single bucket for DGIM Algorithm"""
    
    def __init__(self, timestamp, size):
        self.timestamp = timestamp
        self.size = size
        self.next = None

class DGIM:
    """
    DGIM Algorithm for counting 1s in a sliding window
    Efficient for streaming data with memory constraints
    """
    
    def __init__(self, window_size=3600):  # 1 hour default
        """
        Initialize DGIM Algorithm
        
        Args:
            window_size: Size of sliding window in seconds
        """
        self.window_size = window_size
        self.buckets = {}  # size -> list of buckets
        self.current_time = 0
        self.total_bits = 0
        
        print(f"📊 DGIM Algorithm initialized:")
        print(f"   Window size: {window_size} seconds ({window_size/3600:.1f} hours)")
    
    def _remove_expired_buckets(self):
        """Remove buckets outside the window"""
        cutoff_time = self.current_time - self.window_size
        
        for size in list(self.buckets.keys()):
            bucket_list = self.buckets[size]
            # Remove expired buckets from the end (oldest)
            while bucket_list and bucket_list[-1].timestamp < cutoff_time:
                bucket_list.pop()
            
            # Remove empty size categories
            if not bucket_list:
                del self.buckets[size]
    
    def _merge_buckets(self, size):
        """Merge buckets of same size if there are more than 2"""
        if size not in self.buckets or len(self.buckets[size]) <= 2:
            return
        
        bucket_list = self.buckets[size]
        if len(bucket_list) > 2:
            # Remove the two oldest buckets and create one bucket of double size
            oldest = bucket_list.pop()
            second_oldest = bucket_list.pop()
            
            # Create new bucket with doubled size and newer timestamp
            new_size = size * 2
            new_bucket = DGIMBucket(second_oldest.timestamp, new_size)
            
            if new_size not in self.buckets:
                self.buckets[new_size] = []
            self.buckets[new_size].insert(0, new_bucket)
            
            # Recursively merge if needed
            self._merge_buckets(new_size)
    
    def add_bit(self, bit, timestamp=None):
        """Add a bit to the stream"""
        if timestamp is None:
            timestamp = time.time()
        
        self.current_time = timestamp
        self.total_bits += 1
        
        # Only process 1-bits
        if bit == 1:
            # Create new bucket of size 1
            new_bucket = DGIMBucket(timestamp, 1)
            
            if 1 not in self.buckets:
                self.buckets[1] = []
            
            # Add to front (newest)
            self.buckets[1].insert(0, new_bucket)
            
            # Merge buckets if necessary
            self._merge_buckets(1)
        
        # Clean up expired buckets
        self._remove_expired_buckets()
    
    def count_ones_in_window(self):
        """Count 1s in the current window"""
        self._remove_expired_buckets()
        
        total_count = 0
        
        # Sum all complete buckets
        for size, bucket_list in self.buckets.items():
            if len(bucket_list) > 1:
                # All buckets except the oldest
                total_count += size * (len(bucket_list) - 1)
            
            # For the oldest bucket of each size, add half its value (DGIM approximation)
            if bucket_list:
                total_count += size // 2
        
        return total_count
    
    def get_exact_count_in_window(self, bit_stream):
        """Get exact count for comparison (not part of DGIM, just for testing)"""
        cutoff_time = self.current_time - self.window_size
        count = 0
        
        for timestamp, bit in bit_stream:
            if timestamp >= cutoff_time and timestamp <= self.current_time and bit == 1:
                count += 1
        
        return count
    
    def get_stats(self):
        """Get DGIM statistics"""
        bucket_count = sum(len(bucket_list) for bucket_list in self.buckets.values())
        memory_usage = bucket_count * 16  # Approximate bytes per bucket
        
        return {
            'window_size': self.window_size,
            'current_time': self.current_time,
            'total_bits_processed': self.total_bits,
            'bucket_count': bucket_count,
            'memory_usage_bytes': memory_usage,
            'estimated_ones_in_window': self.count_ones_in_window(),
            'bucket_sizes': list(self.buckets.keys())
        }

class StreamMiningDemo:
    """
    Demonstration of stream mining algorithms for smart meter data
    """
    
    def __init__(self):
        self.bloom_filter = BloomFilter(capacity=10000, error_rate=0.01)
        self.dgim = DGIM(window_size=3600)  # 1 hour window
        self.anomaly_history = []
        
    def process_anomaly_alert(self, building_id, timestamp, anomaly_score):
        """Process an anomaly alert through stream mining algorithms"""
        
        # Create unique identifier for this anomaly
        anomaly_id = f"{building_id}_{timestamp}_{anomaly_score:.3f}"
        
        # Check if this is a duplicate using Bloom Filter
        is_duplicate = self.bloom_filter.contains(anomaly_id)
        
        if not is_duplicate:
            # Add to Bloom Filter
            self.bloom_filter.add(anomaly_id)
            
            # Add to DGIM (1 for anomaly, 0 for normal)
            self.dgim.add_bit(1, timestamp)
            
            # Store for history
            self.anomaly_history.append({
                'building_id': building_id,
                'timestamp': timestamp,
                'anomaly_score': anomaly_score,
                'anomaly_id': anomaly_id
            })
            
            return {
                'is_duplicate': False,
                'anomaly_id': anomaly_id,
                'bloom_filter_stats': self.bloom_filter.get_stats(),
                'dgim_stats': self.dgim.get_stats()
            }
        else:
            return {
                'is_duplicate': True,
                'anomaly_id': anomaly_id,
                'message': 'Duplicate anomaly detected by Bloom Filter'
            }
    
    def get_anomaly_count_in_window(self):
        """Get estimated count of anomalies in current window using DGIM"""
        return self.dgim.count_ones_in_window()
    
    def simulate_stream_processing(self, num_anomalies=1000):
        """Simulate stream processing with random anomalies"""
        
        print("🌊 Simulating Stream Processing...")
        print("=" * 50)
        
        import random
        
        start_time = time.time()
        current_time = start_time
        
        # Statistics
        total_processed = 0
        duplicates_detected = 0
        
        for i in range(num_anomalies):
            # Simulate time progression (random intervals)
            current_time += random.uniform(1, 300)  # 1 second to 5 minutes
            
            # Random building and anomaly score
            building_id = random.randint(1, 200)
            anomaly_score = random.uniform(0.5, 1.0)
            
            # Occasionally create duplicates for testing
            if random.random() < 0.1:  # 10% chance of duplicate
                # Reuse recent anomaly
                if self.anomaly_history:
                    recent = random.choice(self.anomaly_history[-10:])
                    building_id = recent['building_id']
                    anomaly_score = recent['anomaly_score']
            
            # Process through stream mining
            result = self.process_anomaly_alert(building_id, current_time, anomaly_score)
            
            total_processed += 1
            if result['is_duplicate']:
                duplicates_detected += 1
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{num_anomalies} anomalies...")
        
        # Final statistics
        print(f"\n📊 Stream Processing Results:")
        print(f"   Total anomalies processed: {total_processed}")
        print(f"   Duplicates detected: {duplicates_detected}")
        print(f"   Duplicate rate: {duplicates_detected/total_processed*100:.2f}%")
        
        # Bloom Filter stats
        bf_stats = self.bloom_filter.get_stats()
        print(f"\n🔍 Bloom Filter Performance:")
        print(f"   Items added: {bf_stats['items_added']:,}")
        print(f"   Memory utilization: {bf_stats['utilization']*100:.1f}%")
        print(f"   Estimated false positive rate: {bf_stats['estimated_fpr']*100:.3f}%")
        
        # DGIM stats
        dgim_stats = self.dgim.get_stats()
        print(f"\n📊 DGIM Algorithm Performance:")
        print(f"   Estimated anomalies in window: {dgim_stats['estimated_ones_in_window']}")
        print(f"   Memory usage: {dgim_stats['memory_usage_bytes']} bytes")
        print(f"   Bucket count: {dgim_stats['bucket_count']}")
        
        return {
            'total_processed': total_processed,
            'duplicates_detected': duplicates_detected,
            'bloom_filter_stats': bf_stats,
            'dgim_stats': dgim_stats
        }

def demo_stream_mining():
    """Main demo function"""
    
    print("🌊 Smart Meter Stream Mining Demo")
    print("=" * 60)
    print("Demonstrating Big Data stream processing concepts:")
    print("• Bloom Filter: Duplicate anomaly detection")
    print("• DGIM Algorithm: Sliding window anomaly counting")
    print("=" * 60)
    
    # Create demo instance
    demo = StreamMiningDemo()
    
    # Run simulation
    results = demo.simulate_stream_processing(num_anomalies=500)
    
    print(f"\n✅ Demo completed successfully!")
    print(f"💡 Key Insights:")
    print(f"   - Bloom Filter provides memory-efficient duplicate detection")
    print(f"   - DGIM Algorithm enables approximate counting in sliding windows")
    print(f"   - Perfect for real-time anomaly monitoring systems")
    
    return results

if __name__ == "__main__":
    demo_stream_mining()