import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="Chicago Crime Clustering", layout="wide")

# --- Title & Description ---
st.title("🚔 Chicago Crime Clustering & Mapping App")

st.markdown("""
Cluster crime incidents by location and area using machine learning. You can use the default **`chicago_crime_1000.csv`** or upload your own file.

### 📥 Expected CSV Columns

| Column            | Type    | Description                        |
|------------------|---------|------------------------------------|
| `Latitude`        | float   | Crime latitude                     |
| `Longitude`       | float   | Crime longitude                    |
| `Ward`            | int     | Ward number                        |
| `Community Area`  | int     | Community Area number              |
| `Primary Type`    | string  | Crime category (e.g., THEFT)       |

### 📘 Sample CSV
""")

sample_data = {
    "Latitude": [41.8781, 41.8850, 41.8700, 41.8950],
    "Longitude": [-87.6298, -87.6200, -87.6350, -87.6205],
    "Ward": [42, 43, 42, 43],
    "Community Area": [25, 26, 25, 26],
    "Primary Type": ["THEFT", "ASSAULT", "THEFT", "BATTERY"]
}
sample_df = pd.DataFrame(sample_data)
st.download_button("📥 Download Sample CSV", data=sample_df.to_csv(index=False),
                   file_name="sample_crime.csv", mime="text/csv")

# --- File Upload & Fallback ---
uploaded_file = st.file_uploader("Upload your own crime CSV file", type=["csv"])
default_path = "crime-data-1000.csv"

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded.")
elif os.path.exists(default_path):
    df = pd.read_csv(default_path)
    st.info("ℹ️ Using default dataset: `chicago_crime_1000.csv`")
else:
    st.error("❌ No file uploaded and default dataset not found.")
    st.stop()

# --- Clean & Prepare Data ---
required_cols = ['Latitude', 'Longitude', 'Ward', 'Community Area']
df = df.dropna(subset=required_cols)
X = df[required_cols]
idx = X.index
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Sidebar Controls ---
st.sidebar.header("🧪 Clustering Options")
algorithm = st.sidebar.selectbox("Select Algorithm", ['KMeans', 'DBSCAN', 'Agglomerative'])

if algorithm == 'KMeans':
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif algorithm == 'DBSCAN':
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 0.5)
    min_samples = st.sidebar.slider("Min samples", 2, 50, 10)
    model = DBSCAN(eps=eps, min_samples=min_samples)
else:
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)
    model = AgglomerativeClustering(n_clusters=n_clusters)

# --- Apply Clustering ---
labels = model.fit_predict(X_scaled)
df.loc[idx, 'Cluster'] = labels.astype(int)

# --- Assign Meaningful Names ---
if 'Primary Type' in df.columns:
    summary = df.loc[idx].groupby('Cluster')['Primary Type'].agg(lambda x: x.value_counts().idxmax())
    name_map = {cl: f"{crime} Cluster" for cl, crime in summary.items()}
else:
    name_map = {cl: f"Cluster {cl}" for cl in df['Cluster'].unique()}

df['Cluster Name'] = df['Cluster'].map(name_map)

# --- Scoring with human-readable ratings ---
def silhouette_rating(score):
    if score < 0.25:
        return "❌ Poor"
    elif score < 0.5:
        return "⚠️ Moderate"
    elif score < 0.7:
        return "✅ Good"
    else:
        return "🌟 Excellent"

st.subheader("📊 Cluster Evaluation")
unique_labels = set(labels)
noise_present = -1 in unique_labels
valid_clusters = len(unique_labels - {-1}) if noise_present else len(unique_labels)

if valid_clusters > 1:
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    rating = silhouette_rating(sil_score)

    st.markdown(f"""
    - **Silhouette Score:** {sil_score:.3f} → **{rating}**
    - **Davies-Bouldin Index:** {db_score:.2f} (lower = better)
    - **Calinski-Harabasz Score:** {ch_score:.2f} (higher = better)
    """)
else:
    st.warning("⚠️ Clustering didn't produce valid results. Try adjusting parameters.")

# --- PCA Plot ---
st.subheader("📉 PCA 2D Visualization")
X_pca = PCA(n_components=2).fit_transform(X_scaled)
viz_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
viz_df['Cluster'] = df.loc[idx, 'Cluster Name']

fig, ax = plt.subplots()
sns.scatterplot(data=viz_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
st.pyplot(fig)

# --- Folium Map ---
st.subheader("🗺️ Cluster Map")
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()
folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=11)

colors = sns.color_palette("tab10", len(df['Cluster Name'].unique())).as_hex()
color_map = dict(zip(df['Cluster Name'].unique(), colors))

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=color_map.get(row['Cluster Name'], "#000"),
        fill=True,
        fill_opacity=0.7,
        popup=f"Cluster: {row['Cluster Name']}<br>Type: {row.get('Primary Type', 'N/A')}"
    ).add_to(folium_map)

st_folium(folium_map, width=700, height=500)

# --- Preview & Download ---
st.subheader("📋 Preview Clustered Data")
st.dataframe(df.head(10))

st.download_button("📥 Download Clustered CSV", df.to_csv(index=False),
                   file_name="clustered_crime_output.csv", mime="text/csv")
