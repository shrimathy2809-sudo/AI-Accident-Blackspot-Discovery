from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Dataset" / "accident_clean.csv"
MODEL_DIR = ROOT / "Models"
MODEL_PATH = MODEL_DIR / "blackspot_model.pkl"
CLUSTERED_PATH = ROOT / "Dataset" / "accident_with_clusters.csv"
SUMMARY_PATH = ROOT / "Dataset" / "blackspot_summary.csv"
METRICS_PATH = ROOT / "metrics.json"

def train_model():
    df = pd.read_csv(DATA)
    coords = df[["latitude", "longitude"]].to_numpy(dtype=float)

    # DBSCAN uses radians with haversine distance.
    earth_radius_km = 6371.0088
    eps_km = 0.5
    model = DBSCAN(
        eps=eps_km / earth_radius_km,
        min_samples=3,
        metric="haversine",
        algorithm="ball_tree"
    )
    labels = model.fit_predict(np.radians(coords))
    df["cluster"] = labels

    clustered = df[df["cluster"] != -1].copy()
    if clustered.empty:
        raise ValueError(
            "No clusters found. Add more nearby accident records or increase eps_km."
        )

    summary = clustered.groupby("cluster").agg(
        accident_count=("cluster", "size"),
        center_latitude=("latitude", "mean"),
        center_longitude=("longitude", "mean")
    ).reset_index()
    summary["blackspot_level"] = pd.cut(
        summary["accident_count"],
        bins=[0, 4, 9, float("inf")],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    ).astype(str)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "eps_km": eps_km, "min_samples": 3},
        MODEL_PATH
    )
    df.to_csv(CLUSTERED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    metrics = {
        "total_records": int(len(df)),
        "cluster_count": int(summary.shape[0]),
        "noise_records": int((labels == -1).sum()),
        "clustered_records": int((labels != -1).sum()),
        "silhouette_score": None
    }
    valid_labels = set(labels) - {-1}
    if len(valid_labels) >= 2:
        try:
            metrics["silhouette_score"] = float(
                silhouette_score(coords[labels != -1], labels[labels != -1])
            )
        except Exception:
            pass

    METRICS_PATH.write_text(json.dumps(metrics, indent=4))
    print("Training completed.")
    print(json.dumps(metrics, indent=4))
    print(f"Model saved to: {MODEL_PATH}")
    return model, summary, metrics

if __name__ == "__main__":
    train_model()
