from pathlib import Path
import mlflow
from train import train_model

ROOT = Path(__file__).resolve().parent

def track_training():

    # Use SQLite database instead of old file store
    db_path = ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("Accident_Blackspot_Discovery")

    with mlflow.start_run():

        model, summary, metrics = train_model()

        mlflow.log_param("algorithm", "DBSCAN")
        mlflow.log_param("eps_km", 0.5)
        mlflow.log_param("min_samples", 3)

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, value)

        mlflow.log_artifact(
            str(ROOT / "Dataset" / "blackspot_summary.csv")
        )

        mlflow.log_artifact(
            str(ROOT / "Models" / "blackspot_model.pkl")
        )

        print("MLflow tracking completed successfully!")

if __name__ == "__main__":
    track_training()