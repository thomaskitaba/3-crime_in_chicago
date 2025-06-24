#!/usr/bin/python3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import MarkerCluster

# --- Step 1: Load data ---
df = pd.read_csv("crime-data-1000.csv")

# --- Step 2: Drop rows missing Latitude or Longitude ---
df = df.dropna(subset=['Latitude', 'Longitude'])

# --- Step 3: Prepare data for clustering (latitude and longitude) ---
X = df[['Latitude', 'Longitude']].values

# Optional: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 4: Apply KMeans clustering ---
k = 4  # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Assign cluster labels to DataFrame
df['Cluster'] = labels

# --- Step 5: Determine dominant crime type per cluster ---
cluster_crime_dist = (
    df.groupby(['Cluster', 'Primary Type'])
      .size()
      .unstack(fill_value=0)
      .apply(lambda x: x / x.sum(), axis=1)
)

def get_dominant_crime(cluster):
    return cluster_crime_dist.loc[cluster].idxmax()

cluster_names = {cluster: get_dominant_crime(cluster) for cluster in cluster_crime_dist.index}
df['Cluster Name'] = df['Cluster'].map(cluster_names)

print("Cluster names assigned:")
print(cluster_names)

# --- Step 6: Plot clusters on Folium map ---
chicago_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
marker_cluster = MarkerCluster().add_to(chicago_map)

# Define colors for clusters (up to 10)
cluster_colors = [
    'red', 'blue', 'green', 'purple', 'orange',
    'darkred', 'lightblue', 'lightgreen', 'beige', 'gray'
]

for _, row in df.iterrows():
    cluster_num = int(row['Cluster'])
    color = cluster_colors[cluster_num % len(cluster_colors)]
    popup_text = f"Cluster: {row['Cluster Name']}<br>Crime: {row['Primary Type']}"
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=popup_text
    ).add_to(marker_cluster)

# --- Step 7: Save map ---
map_filename = "crime_clusters_map.html"
chicago_map.save(map_filename)
print(f"Map saved to '{map_filename}'. Open it in your browser.")
