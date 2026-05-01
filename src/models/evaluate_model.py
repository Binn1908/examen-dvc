import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed_data")
MODELS_PATH = Path("models")
METRICS_PATH = Path("metrics")


def evaluate_model():
    X_test = pd.read_csv(PROCESSED_DATA_PATH / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DATA_PATH / "y_test.csv").squeeze()

    with open(MODELS_PATH / "trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)

    pd.DataFrame(predictions, columns=["predicted_silica_concentrate"]).to_csv(
        PROCESSED_DATA_PATH / "predictions.csv", index=False
    )

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    scores = {"mse": mse, "r2": r2}
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH / "scores.json", "w") as f:
        json.dump(scores, f)
    print("Evaluation complete. Scores saved successfully.")


if __name__ == "__main__":
    evaluate_model()
