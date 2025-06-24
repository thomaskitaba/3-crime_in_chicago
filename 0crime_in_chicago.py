#!/usr/bin/python3 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

    # 4. Apply KMeans
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # 5. Assign labels to original DataFrame
    df.loc[X_index, 'Cluster'] = labels

    # 6. Evaluation Metrics
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

    # 7. Get cluster composition by crime type
    if 'Primary Type' in df.columns:
        cluster_summary = df.loc[X_index].groupby('Cluster')['Primary Type'].value_counts(normalize=True).unstack().fillna(0)
        
        # Find dominant crime in each cluster
        top_crimes = cluster_summary.idxmax(axis=1)
        print("\nTop Primary Crime per Cluster:")
        print(top_crimes)

        # Map cluster numbers to names (customize if needed)
        cluster_names = {
            0: 'Theft-heavy Area',
            1: 'Battery-heavy Area',
            2: 'Narcotics Zone',
            3: 'Assault Zone'
        }

        # Apply mapping
        df.loc[X_index, 'Cluster Name'] = df.loc[X_index, 'Cluster'].map(cluster_names)
        cluster_summary.index = cluster_summary.index.map(cluster_names)
    else:
        print("\nPrimary Type not found, skipping naming and crime breakdown")
        df.loc[X_index, 'Cluster Name'] = df.loc[X_index, 'Cluster'].astype(str)
        cluster_summary = None

    # 8. PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=df.loc[X_index, 'Cluster Name'],
        palette="Set2"
    )
