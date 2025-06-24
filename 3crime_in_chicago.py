#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import folium
from folium.plugins import MarkerCluster

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

    # 4. Create connectivity matrix using KNN graph
    connectivity = kneighbors_graph(X_scaled, n_neighbors=10, include_self=False)

    # 5. Apply Agglomerative Clustering
    k = 4
    model = AgglomerativeClustering(n_clusters=k, connectivity=connectivity, linkage='ward')
    labels = model.fit_predict(X_scaled)

    # 6. Assign labels to original DataFrame
    df.loc[X_index, 'Cluster'] = labels

    # 7. Evaluation Metrics
    sil_score = silhouette_score(X_scaled, labels)
    dbi_score = davies_bouldin_score(X_scaled, labels)
    chi_score = calinski_harabasz_score(X_scaled, labels)

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

    print("\n📊 Clustering Evaluation Metrics:")
    print(f"• Silhouette Score: {sil_score:.3f} → {interpret_score(sil_score, 'silhouette')}")
    print(f"• Davies-Bouldin Index: {dbi_score:.3f} → {interpret_score(dbi_score, 'dbi')}")
    print(f"• Calinski-Harabasz Score: {chi_score:.0f} → {interpret_score(chi_score, 'chi')}")

    # 8. Get cluster composition by crime type
    if 'Primary Type' in df.columns:
        cluster_summary = df.loc[X_index].groupby('Cluster')['Primary Type'].value_counts(normalize=True).unstack().fillna(0)
        
        top_crimes = cluster_summary.idxmax(axis=1)
        print("\nTop Primary Crime per Cluster:")
        print(top_crimes)

        cluster_names = {
            0: 'Theft-heavy Area',
            1: 'Battery-heavy Area',
            2: 'Narcotics Zone',
            3: 'Assault Zone'
        }

        df.loc[X_index, 'Cluster Name'] = df.loc[X_index, 'Cluster'].map(cluster_names)
        cluster_summary.index = cluster_summary.index.map(cluster_names)
    else:
        print("\nPrimary Type not found, skipping naming and crime breakdown")
        df.loc[X_index, 'Cluster Name'] = df.loc[X_index, 'Cluster'].astype(str)
        cluster_summary = None

    # 9. PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df.loc[X_index, 'Cluster Name'],
        palette="Set2"
    )
    plt.title("PCA Projection of Crime Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    # 10. t-SNE Visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=df.loc[X_index, 'Cluster Name'],
        palette="Set1"
    )
    plt.title("t-SNE Projection of Crime Clusters")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    # 11. Cluster Composition Plot
    if cluster_summary is not None:
        cluster_summary.plot(
            kind='bar',
            stacked=True,
            figsize=(10, 6),
            colormap="tab20"
        )
        plt.title("Crime Type Distribution by Cluster")
        plt.ylabel("Proportion")
        plt.xlabel("Cluster")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Crime Type')
        plt.tight_layout()
        plt.show()

    # 12. Folium Map
    map_df = df.loc[X_index].dropna(subset=['Latitude', 'Longitude', 'Cluster'])

    chicago_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
    marker_cluster = MarkerCluster().add_to(chicago_map)

    cluster_colors = [
        'red', 'blue', 'green', 'purple', 'orange',
        'darkred', 'lightblue', 'lightgreen', 'beige', 'gray'
    ]

    for _, row in map_df.iterrows():
        cluster_num = int(row['Cluster'])
        color = cluster_colors[cluster_num % len(cluster_colors)]
        popup_text = f"Cluster: {row['Cluster Name']}<br>Crime: {row.get('Primary Type', 'Unknown')}"
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(marker_cluster)

    map_filename = "crime_clusters_map.html"
    chicago_map.save(map_filename)
    print(f"\nMap saved to '{map_filename}'. Open it in your browser to explore clusters.")
