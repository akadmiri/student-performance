import matplotlib.pyplot as plt
import pandas as pd
import prince
import seaborn as sns

path = "data/raw/student-mat.csv"
df = pd.read_csv(path, sep=";")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

# Histogramme avec courbe de densité
sns.histplot(data=df, x="G3", kde=True, bins=20, color="blue")
plt.title("Distribution des notes finales (G3)", fontsize=14, fontweight="bold")
plt.xlabel("Note finale (G3)/20", fontsize=10)
plt.ylabel("Nombre d'étudiants", fontsize=10)
plt.tight_layout()

# Enregistrement de l'image
output_image_path = "output/distribution_G3.png"
plt.savefig(output_image_path, dpi=300)

plt.show()

# Visualisation des variables numériques:
num_df = df.select_dtypes(include=["int64"])
corr_matrice = num_df.corr()

plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrice, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
plt.title(
    "Matrice de corrélation des variables numériques", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("output/matrice_correlation.png", dpi=300)
plt.show()

# Visualisation des variables catégorielles: mca
cat_df = df.select_dtypes(include=["str"])

mca = prince.MCA(n_components=2, random_state=6)
mca = mca.fit(cat_df)
cordonnees = mca.column_coordinates(cat_df)

plt.figure(figsize=(14, 8))
sns.scatterplot(x=cordonnees[0], y=cordonnees[1], s=100, color="red")

for index, row in cordonnees.iterrows():
    plt.text(row[0] + 0.02, row[1], str(index), fontsize=9, alpha=0.8, weight="bold")

plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
plt.title(
    "Analyse des correspondances multiples (MCA) des variables catégorielles",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Dimension 1", fontsize=10)
plt.ylabel("Dimension 2", fontsize=10)
plt.tight_layout()
plt.savefig("output/mca_categorielle.png", dpi=300)
plt.show()


# Boxplot de G3 en fonction de la variable 'higher' (aspiration à l'enseignement supérieur)
plt.figure(figsize=(12, 8))
sns.boxplot(x="higher", y="G3", data=df, palette="Set2")
plt.title(
    "Impact de l'aspiration à l'enseignement supérieur sur les notes finales",
    fontsize=14,
)
plt.xlabel("Souhaite poursuivre des études supérieures", fontsize=10)
plt.ylabel("Note finale (G3)/20", fontsize=10)
plt.tight_layout()
plt.savefig("output/boxplot_Sup_G3.png", dpi=300)
plt.show()

# Barplot de G3 en fonction de la variable
plt.figure(figsize=(12, 8))
sns.barplot(x="schoolsup", y="G3", data=df, hue="sex", errorbar=None, palette="Pastel1")
plt.title("Impact du soutien scolaire sur les notes finales", fontsize=14)
plt.xlabel("Soutien scolaire", fontsize=10)
plt.ylabel("Note moyenne (G3)/20", fontsize=10)
plt.legend(title="Sexe")
plt.tight_layout()
plt.savefig("output/barplot_Soutien_G3.png", dpi=300)
plt.show()
