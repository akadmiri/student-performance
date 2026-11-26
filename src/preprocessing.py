from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "student-mat.csv"


def load_data():
    return pd.read_csv(DATA_PATH, sep=";")


def preprocess_data(df):
    X = df.drop(columns=["G3"])
    Y = df["G3"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=6
    )

    variables_num = X_train.select_dtypes(include="number").columns.tolist()
    variables_cat = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), variables_num),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                variables_cat,
            ),
        ]
    )

    # Fit the scaler and encoder on the training rows only.
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    print(f"Training set after preprocessing: {X_train_preprocessed.shape}")
    print(f"Test set after preprocessing: {X_test_preprocessed.shape}")

    return X_train_preprocessed, X_test_preprocessed, Y_train, Y_test


if __name__ == "__main__":
    preprocess_data(load_data())
