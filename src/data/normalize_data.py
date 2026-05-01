import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed_data")


def normalize_data():
    X_train = pd.read_csv(PROCESSED_DATA_PATH / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_PATH / "X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        PROCESSED_DATA_PATH / "X_train_scaled.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
        PROCESSED_DATA_PATH / "X_test_scaled.csv", index=False
    )

    print("Data normalized successfully.")


if __name__ == "__main__":
    normalize_data()
