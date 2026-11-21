from pathlib import Path
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
CLUSTERED = ROOT / "Dataset" / "accident_with_clusters.csv"
SUMMARY = ROOT / "Dataset" / "blackspot_summary.csv"

st.set_page_config(page_title="Accident Blackspot Dashboard", layout="wide")
st.title("AI-Based Hidden Accident Blackspot Discovery")
st.write("Map of accident records and DBSCAN-discovered clusters.")

if not CLUSTERED.exists() or not SUMMARY.exists():
    st.warning("Run: python clean_data.py, then python train.py")
    st.stop()

df = pd.read_csv(CLUSTERED)
summary = pd.read_csv(SUMMARY)
a, b = st.columns(2)
a.metric("Accident records", len(df))
b.metric("Discovered clusters", len(summary))

center = [df["latitude"].mean(), df["longitude"].mean()]
map_view = folium.Map(location=center, zoom_start=11)
for _, row in df.iterrows():
    color = "gray" if int(row["cluster"]) == -1 else "red"
    folium.CircleMarker(
        [row["latitude"], row["longitude"]],
        radius=4, color=color, fill=True,
        popup=f"Cluster: {row['cluster']}"
    ).add_to(map_view)

for _, row in summary.iterrows():
    folium.Marker(
        [row["center_latitude"], row["center_longitude"]],
        popup=f"Cluster {row['cluster']} | Accidents: {row['accident_count']} | Level: {row['blackspot_level']}",
        icon=folium.Icon(color="red", icon="warning-sign")
    ).add_to(map_view)

st_folium(map_view, width=1100, height=550)
st.subheader("Blackspot summary")
st.dataframe(summary, use_container_width=True)
