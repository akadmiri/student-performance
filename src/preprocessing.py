import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

path = "data/raw/student-mat.csv"
df = pd.read_csv(path, sep=";")

# Echantillon:
print("Echantillon:")
print(df.head())


# Variables cibles et explicatives:
X = df.drop(columns=["G3"])
Y = df["G3"]

# Séparation des données en ensembles d'entraînement et de test:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=6)

print(f"Donnees d'entraînement: {X_train.shape}")
print(f"Donnees de test: {X_test.shape}")

# types de données:
print("types de données:")
print(df.dtypes)

variables_num = X_train.select_dtypes(include=["int64"]).columns.tolist()
variables_cat = X_train.select_dtypes(include=["str"]).columns.tolist()
print(f"Nombre de variables numériques : {len(variables_num)}")
print(f"Nombre de variables catégorielles : {len(variables_cat)}")

# Prétraitement des données:
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
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

print(f"Forme des données d'entraînement prétraitées: {X_train_preprocessed.shape}")
print(f"Forme des données de test prétraitées: {X_test_preprocessed.shape}")
