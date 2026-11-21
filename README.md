# AI-Based Hidden Accident Blackspot Discovery using Machine Learning

## What it does
Uses DBSCAN clustering on accident latitude/longitude coordinates to identify dense accident-location clusters.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Requirements.txt
```

## Run in this order
```powershell
python clean_data.py
python train.py
python Predict.py
python app.py
```

Open http://127.0.0.1:5000

Dashboard (separate terminal):
```powershell
streamlit run dashboard.py
```

MLflow:
```powershell
python mlflow_tracking.py
mlflow ui --backend-store-uri ./mlruns
```
Open http://127.0.0.1:5000 for Flask. MLflow UI normally uses port 5000 by default; if Flask is running, use:
```powershell
mlflow ui --backend-store-uri ./mlruns --port 5001
```

Docker:
```powershell
docker build -t accident-blackspot .
docker run -p 5000:5000 accident-blackspot
```

## Dataset columns
Required: latitude, longitude. Optional: date, severity, weather, road_condition, traffic_volume.
The included CSV is synthetic sample data for testing only. Replace it with verified real accident records before drawing real-world conclusions.

## Important limitation
DBSCAN identifies spatially dense groups in the supplied records. It does not prove that a location is dangerous or predict future accidents. Real deployment needs representative, time-stamped data, exposure/traffic-volume normalization, validation, and domain review. `Predict.py` uses distance to learned DBSCAN core points as a practical nearby-cluster check because DBSCAN does not provide a native predict method.
