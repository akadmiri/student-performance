from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import X_test_preprocessed, X_train_preprocessed, Y_test, Y_train


def train_and_evaluate() -> Dict[str, float]:
    model = LinearRegression()
    model.fit(X_train_preprocessed, Y_train)
    y_pred = model.predict(X_test_preprocessed)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(Y_test, y_pred))

    print(
        f"RMSE: {rmse:.4f} points de note (écart moyen entre les notes prédites et réelles)"
    )
    print(f"R²: {r2:.4f} (pourcentage de variance expliquée par le modèle)")

    return {"RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    metrics = train_and_evaluate()
