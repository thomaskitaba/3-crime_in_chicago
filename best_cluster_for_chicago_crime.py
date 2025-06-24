#!/usr/bin/python3
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

# --- Title & Explanation ---
st.title("🚔 Crime Clustering & Naming App")

st.markdown("""
This app clusters Chicago crime data based on location and area information, and assigns meaningful cluster names based on the most common crime type.

### Expected CSV format

Your CSV file should contain **at least** the following columns:

| Column Name       | Data Type        | Description                         |
|-------------------|------------------|-----------------------------------|
| `Latitude`        | float            | Latitude coordinate of the crime  |
| `Longitude`       | float            | Longitude coordinate of the crime |
| `Ward`            | integer or float | Ward number                       |
| `Community Area`  | integer or float | Community Area number             |
| `Primary Type`    | string           | Crime type (e.g., THEFT, ASSAULT) |

### Sample CSV

You can download a sample CSV file below to see the format expected by the app.
""")

# --- Sample CSV creation ---
sample_data = {
    "Latitude": [41.8781, 41.8850, 41.8700, 41.8950],
    "Longitude": [-87.6298, -87.6200, -87.6350, -87.6205],
    "Ward": [42, 43, 42, 43],
    "Community Area": [25, 26, 25, 26],
    "Primary Type": ["THEFT", "ASSAULT", "THEFT", "BATTERY"]
}
sample_df = pd.DataFrame(sample_data)
sample_csv = sample_df.to_csv(index=False)

st.download_button(
    label="📥 Download Sample CSV",
    data=sample_csv,
    file_name="sample_chicago_crime.csv",
    mime="text/csv"
)

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your crime CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    features = ['Latitude', 'Longitude', 'Ward', 'Community Area']
    df = df.dropna(subset=features)
    X = df[features]
    idx = X.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Sidebar UI ---
    st.sidebar.header("Clustering Options")
    algorithm = st.sidebar.selectbox("Algorithm", ['KMeans', 'DBSCAN', 'Agglomerative'], index=0)

    if algorithm == 'KMeans':
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == 'DBSCAN':
        eps = st.sidebar.slider("eps", 0.1, 5.0, 0.7)
        min_samples = st.sidebar.slider("min_samples", 2, 50, 10)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)
        model = AgglomerativeClustering(n_clusters=n_clusters)

    # --- Cluster Assignment ---
    labels = model.fit_predict(X_scaled)
    df.loc[idx, 'Cluster'] = labels.astype(int)

    # --- Generate Cluster Names Based on Top Crime ---
    if 'Primary Type' in df.columns:
        summary = df.loc[idx].groupby('Cluster')['Primary Type']\
                      .apply(lambda x: x.value_counts().idxmax())
        name_map = {cl: crime for cl, crime in summary.items()}
    else:
        name_map = {cl: f"Cluster {cl}" for cl in df['Cluster'].unique()}

    df['Cluster Name'] = df['Cluster'].map(name_map).astype(str)

    # --- Define performance rating based on silhouette score ---
    def silhouette_rating(score):
        if score < 0.25:
            return "Poor"
        elif score < 0.50:
            return "Moderate"
        elif score < 0.70:
            return "Good"
        else:
            return "Excellent"

    # --- Display Metrics ---
    st.subheader("📊 Cluster Evaluation Metrics")
    unique_labels = set(labels)
    noise_present = -1 in unique_labels
    valid_clusters = len(unique_labels - {-1}) if noise_present else len(unique_labels)

    if valid_clusters > 1:
        sil_score = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
        
        rating = silhouette_rating(sil_score)
        
        st.write(f"**Silhouette Score:** {sil_score:.3f} → **{rating}**")
        st.write(f"**Davies-Bouldin Index:** {db_score:.3f} (Lower is better)")
        st.write(f"**Calinski-Harabasz Score:** {ch_score:.0f} (Higher is better)")
    else:
        st.warning("Too few clusters or DBSCAN created mostly noise. Adjust parameters and try again.")

    # --- PCA Plot ---
    st.subheader("📉 PCA Visualization")
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    viz = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    viz['Cluster Name'] = df.loc[idx, 'Cluster Name']
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=viz, x="PC1", y="PC2", hue="Cluster Name", palette="tab10", ax=ax)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    st.pyplot(fig)

    # --- Folium Map ---
    st.subheader("🗺️ Crime Clusters Map")
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    unique_clusters = df['Cluster Name'].unique()
    palette = sns.color_palette("tab10", len(unique_clusters)).as_hex()
    color_map = dict(zip(unique_clusters, palette))

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=4,
            color=color_map.get(row['Cluster Name'], "#000000"),
            fill=True,
            fill_opacity=0.7,
            popup=f"Cluster: {row['Cluster Name']}<br>Crime: {row.get('Primary Type', 'N/A')}"
        ).add_to(folium_map)

    st_folium(folium_map, width=700, height=500)

    # --- Data & Download ---
    st.subheader("📋 Sample of Clustered Data")
    st.dataframe(df.head())

    csv = df.to_csv(index=False)
    st.download_button("📥 Download Labeled Data", csv, "clustered_crime.csv", "text/csv")

else:
    st.info("⚠️ Please upload a crime CSV to begin.")

