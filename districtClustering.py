import pandas as pd
from collections import deque

def cluster_districts(districts, max_cost_diff=5000, max_aqi_diff=20):
    """Cluster districts using a modified divide-and-conquer approach"""
    if not districts:
        return []
    
    # Sort districts by Total Monthly Cost
    districts_sorted = sorted(districts, key=lambda x: x['Total Monthly Cost'])
    
    clusters = []
    current_cluster = [districts_sorted[0]]
    
    for i in range(1, len(districts_sorted)):
        prev_district = districts_sorted[i-1]
        curr_district = districts_sorted[i]
        
        # Check if should be in same cluster
        cost_diff = curr_district['Total Monthly Cost'] - prev_district['Total Monthly Cost']
        aqi_diff = abs(curr_district['AQI'] - prev_district['AQI'])
        
        if cost_diff <= max_cost_diff and aqi_diff <= max_aqi_diff:
            current_cluster.append(curr_district)
        else:
            clusters.append(current_cluster)
            current_cluster = [curr_district]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters

# Load and prepare data
df = pd.read_csv('gujarat_data.csv')
df['Total Monthly Cost'] = df['Avg Rent (1BHK) (₹)'] + df['Utilities (₹)'] + df['Groceries (₹)']
districts = df.to_dict('records')

# Cluster with parameters
clusters = cluster_districts(districts, max_cost_diff=3000, max_aqi_diff=15)

# Print results
print(f"\nClustered {len(districts)} districts into {len(clusters)} groups:")
for i, cluster in enumerate(clusters, 1):
    avg_cost = sum(d['Total Monthly Cost'] for d in cluster) / len(cluster)
    avg_aqi = sum(d['AQI'] for d in cluster) / len(cluster)
    print(f"\nCluster {i} (Size: {len(cluster)}):")
    print(f"Avg Cost: ₹{avg_cost:,.0f} | Avg AQI: {avg_aqi:.1f}")
    print("Districts:", ", ".join(d['District'] for d in cluster))