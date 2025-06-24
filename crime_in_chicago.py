#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Load your dataset
df = pd.read_csv("crime-data-1000.csv")  # Replace with your actual file name

# 2. Select features for clustering
features = ['Latitude', 'Longitude', 'Ward', 'Community Area']
X = df[features].dropna()

# 3. Preserve index to map clusters back to original df
X_index = X.index

# 4. Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 6. Assign cluster labels to filtered rows in df
df.loc[X_index, 'Cluster'] = labels

# 7. Silhouette score
score = silhouette_score(X_scaled, labels)
print(f"\nSilhouette Score: {score:.3f}")

# 8. PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2")
plt.title("PCA Projection of Clusters")
plt.show()

# 9. t-SNE Visualization (optional - slower)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette="Set1")
plt.title("t-SNE Projection of Clusters")
plt.show()

# 10. Cluster composition by Primary Type
if 'Primary Type' in df.columns:
    cluster_summary = df.loc[X_index].groupby('Cluster')['Primary Type'].value_counts(normalize=True).unstack().fillna(0)
    print("\nPrimary Crime Type Distribution per Cluster:")
    print(cluster_summary)

    # Visualize cluster composition
    cluster_summary.plot(kind='bar', stacked=True, figsize=(10, 6), colormap="tab20")
    plt.title("Crime Type Distribution by Cluster")
    plt.ylabel("Proportion")
    plt.xlabel("Cluster")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("\n'Primary Type' column not found — skipping composition analysis.")

