from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import load_data, preprocess_data
from visualization import plot_results


def train_and_evaluate(df=None) -> Dict[str, float]:
    if df is None:
        df = load_data()

    X_train_preprocessed, X_test_preprocessed, Y_train, Y_test = preprocess_data(df)

    model = LinearRegression()
    model.fit(X_train_preprocessed, Y_train)
    y_pred = model.predict(X_test_preprocessed)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(Y_test, y_pred))

    print(f"Test RMSE: {rmse:.4f} grade points")
    print(f"Test R²: {r2:.4f}")

    return {"RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    df = load_data()
    train_and_evaluate(df)
    plot_results(df)
