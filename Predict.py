from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Models" / "blackspot_model.pkl"

def predict_location(latitude, longitude):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    point = np.radians([[float(latitude), float(longitude)]])
    cluster = int(model.fit_predict(point)[0]) if False else None

    # DBSCAN has no native predict() method. Compare the new point
    # against the training core samples using the same haversine radius.
    from sklearn.metrics.pairwise import haversine_distances
    core = model.components_
    if len(core) == 0:
        return {"is_blackspot": False, "message": "No learned clusters found."}

    distances_km = haversine_distances(point, core)[0] * 6371.0088
    nearest_km = float(distances_km.min())
    is_blackspot = nearest_km <= float(saved["eps_km"])
    return {
        "is_blackspot": bool(is_blackspot),
        "nearest_core_point_distance_km": round(nearest_km, 3),
        "message": "Near a discovered accident cluster" if is_blackspot
                   else "No discovered cluster nearby"
    }

if __name__ == "__main__":
    lat = float(input("Enter latitude: "))
    lon = float(input("Enter longitude: "))
    print(predict_location(lat, lon))
