import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('cleaned_gujarat_data.csv')

# Normalize data for clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df[['Total Monthly Cost', 'AQI', 'Green Cover (%)']])

# Cluster districts into 3 groups
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Total Monthly Cost'], df['AQI'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Total Monthly Cost')
plt.ylabel('AQI')
plt.title('District Clusters by Cost & Air Quality')
for i, txt in enumerate(df['District']):
    plt.annotate(txt, (df['Total Monthly Cost'][i], df['AQI'][i]), fontsize=8)
plt.colorbar(label='Cluster Group')
plt.savefig('clusters.png')
plt.show()