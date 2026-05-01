import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed_data")
MODELS_PATH = Path("models")


def train_model():
    X_train = pd.read_csv(PROCESSED_DATA_PATH / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_PATH / "y_train.csv").squeeze()

    with open(MODELS_PATH / "best_params.pkl", "rb") as f:
        best_params = pickle.load(f)

    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    with open(MODELS_PATH / "trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved successfully.")


if __name__ == "__main__":
    train_model()
