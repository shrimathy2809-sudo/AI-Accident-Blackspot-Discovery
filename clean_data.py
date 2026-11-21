from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Dataset" / "accident_data.csv"
OUTPUT = ROOT / "Dataset" / "accident_clean.csv"

def clean_dataset():
    df = pd.read_csv(SOURCE)
    df = df.drop_duplicates()
    required = ["latitude", "longitude"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df = df[df["latitude"].between(-90, 90)]
    df = df[df["longitude"].between(-180, 180)]

    if "severity" in df.columns:
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
        df["severity"] = df["severity"].fillna(df["severity"].median())

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = df["date"].dt.hour.fillna(0).astype(int)

    OUTPUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Cleaned dataset saved: {OUTPUT}")
    print(f"Rows after cleaning: {len(df)}")
    return df

if __name__ == "__main__":
    clean_dataset()
