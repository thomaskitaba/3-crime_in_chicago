#!/usr/bin/python3 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import folium
from folium.plugins import MarkerCluster

# Helper function to interpret clustering metric scores
def interpret_score(score, metric):
    if metric == "silhouette":
        if score > 0.75: return "Excellent"
        elif score > 0.5: return "Good"
        elif score > 0.25: return "Moderate"
        elif score > 0.0: return "Poor"
        else: return "Not Applicable"
    elif metric == "dbi":
        if score < 0.5: return "Excellent"
        elif score < 1.0: return "Good"
        elif score < 2.0: return "Moderate"
        else: return "Poor"
    elif metric == "chi":
        if score > 1000: return "Excellent"
        elif score > 500: return "Good"
        elif score > 100: return "Moderate"
        else: return "Poor"

# 1. Load data
df = pd.read_csv("crime-data-1000.csv")  # Replace with your actual file name

# 2. Select features for clustering
features = [['Latitude', 'Longitude', 'Ward', 'Community Area']]

for i in range(len(features)):
    X = df[features[i]].dropna()
    X_index = X.index  # Keep track of filtered row indices

    # 3. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Apply DBSCAN
    dbscan = DBSCAN(eps=0.7, min_samples=10)  # Tune eps/min_samples for your data
    labels = dbscan.fit_predict(X_scaled)

    df.loc[X_index, 'Cluster'] = labels

    # 5. Evaluate clustering
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    print(f"\nDBSCAN found {n_clusters} clusters (excluding noise).")

    if n_clusters > 1:
        sil_score = silhouette_score(X_scaled, labels)
        dbi_score = davies_bouldin_score(X_scaled, labels)
        chi_score = calinski_harabasz_score(X_scaled, labels)

        print("\n📊 Clustering Evaluation Metrics:")
        print(f"• Silhouette Score: {sil_score:.3f} → {interpret_score(sil_score, 'silhouette')}")
        print(f"• Davies-Bouldin Index: {dbi_score:.3f} → {interpret_score(dbi_score, 'dbi')}")
        print(f"• Calinski-Harabasz Index: {chi_score:.0f} → {interpret_score(chi_score, 'chi')}")
    else:
        print("\n⚠️ Too few clusters found. Try adjusting DBSCAN eps/min_samples.")

    # 6. Get cluster composition by crime type
    if 'Primary Type' in df.columns:
        cluster_summary = df.loc[X_index].groupby('Cluster')['Primary Type'].value_counts(normalize=True).unstack().fillna(0)
        top_crimes = cluster_summary.idxmax(axis=1)
        print("\nTop Primary Crime per Cluster:")
        print(top_crimes)

        # Map cluster numbers to readable names, including noise (-1)
        cluster_names = {cl: f"Cluster {cl}" if cl != -1 else "Noise" for cl in unique_labels}
        df.loc[X_index, 'Cluster Name'] = df.loc[X_index, 'Cluster'].map(cluster_names)
        cluster_summary.index = cluster_summary.index.map(cluster_names)
    else:
        print("\nPrimary Type not found, skipping naming and crime breakdown")
        df.loc[X_index, 'Cluster Name'] = df.loc[X_index, 'Cluster'].astype(str)
        cluster_summary = None

    # 7. PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=df.loc[X_index, 'Cluster Name'],
        palette='tab10'
    )
    plt.title("PCA Projection of DBSCAN Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    # 8. t-SNE Visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=df.loc[X_index, 'Cluster Name'],
        palette='Set1'
    )
    plt.title("t-SNE Projection of DBSCAN Clusters")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    # 9. Cluster Composition Plot (Stacked Bar)
    if cluster_summary is not None:
        cluster_summary.plot(
            kind='bar',
            stacked=True,
            figsize=(10, 6),
            colormap="tab20"
        )
        plt.title("Crime Type Distribution by DBSCAN Cluster")
        plt.ylabel("Proportion")
        plt.xlabel("Cluster")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Crime Type')
        plt.tight_layout()
        plt.show()

    # 10. Plot clusters on Folium map
    map_df = df.loc[X_index].dropna(subset=['Latitude', 'Longitude', 'Cluster'])

    chicago_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
    marker_cluster = MarkerCluster().add_to(chicago_map)

    cluster_colors = [
        'red', 'blue', 'green', 'purple', 'orange',
        'darkred', 'lightblue', 'lightgreen', 'beige', 'gray'
    ]

    for _, row in map_df.iterrows():
        cluster_num = int(row['Cluster'])
        color = 'black' if cluster_num == -1 else cluster_colors[cluster_num % len(cluster_colors)]
        popup_text = f"Cluster: {row['Cluster Name']}<br>Crime: {row.get('Primary Type', 'Unknown')}"
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(marker_cluster)

    map_filename = "crime_clusters_dbscan.html"
    chicago_map.save(map_filename)
    print(f"\nDBSCAN map saved to '{map_filename}'. Open it in your browser to explore clusters.")
