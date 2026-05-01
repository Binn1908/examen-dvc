import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed_data")
MODELS_PATH = Path("models")


def run_grid_search():
    X_train = pd.read_csv(PROCESSED_DATA_PATH / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_PATH / "y_train.csv").squeeze()
    # squeeze() transforms the DataFrame column into a Series column

    param_grid = {
        "n_estimators": [100, 200, 300],  # number of trees in the forest
        "max_depth": [None, 10, 20, 30],  # depth of each tree
        "min_samples_split": [
            2,
            5,
            10,
        ],  # minimum number of samples required to split a node
        "min_samples_leaf": [1, 2, 4],  # minimum samples required at a leaf node
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="r2", n_jobs=-1)
    # The r2 score measures how well the model explains the variance in silica_concentrate
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    with open(MODELS_PATH / "best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    print("Best parameters saved successfully.")


if __name__ == "__main__":
    run_grid_search()
