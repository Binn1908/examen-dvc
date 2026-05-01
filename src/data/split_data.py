import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import requests

RAW_DATA_PATH = Path("data/raw_data")
PROCESSED_DATA_PATH = Path("data/processed_data")


def download_data():
    url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    raw_file = RAW_DATA_PATH / "raw.csv"
    with open(raw_file, "wb") as f:
        f.write(response.content)
    print("Data downloaded successfully.")
    return raw_file


def split_data(raw_file):
    df = pd.read_csv(raw_file)

    # Extract datetime features
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df = df.drop(columns=["date"])

    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DATA_PATH / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_PATH / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_PATH / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_PATH / "y_test.csv", index=False)
    print("Data split successfully.")


if __name__ == "__main__":
    raw_file = download_data()
    split_data(raw_file)
