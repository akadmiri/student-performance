import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import prince
import seaborn as sns

from preprocessing import load_data

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def plot_results(df, show=False):
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))

    # Final grade distribution
    sns.histplot(data=df, x="G3", kde=True, discrete=True, color="blue")
    plt.title("Distribution of final grades", fontsize=14, fontweight="bold")
    plt.xlabel("Final grade (G3, out of 20)", fontsize=10)
    plt.ylabel("Number of students", fontsize=10)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "distribution_G3.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # Pearson correlations for numeric variables
    num_df = df.select_dtypes(include="number")
    corr_matrix = num_df.corr()

    plt.figure(figsize=(14, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title(
        "Pearson correlations for numeric variables", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "matrice_correlation.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # Multiple correspondence analysis of categorical variables
    cat_df = df.select_dtypes(include=["object", "string"])

    mca = prince.MCA(n_components=2, random_state=6)
    mca = mca.fit(cat_df)
    coordinates = mca.column_coordinates(cat_df)

    plt.figure(figsize=(14, 8))
    sns.scatterplot(x=coordinates[0], y=coordinates[1], s=100, color="red")

    for index, row in coordinates.iterrows():
        plt.text(row[0] + 0.02, row[1], str(index), fontsize=9, alpha=0.8, weight="bold")

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.title(
        "Multiple correspondence analysis of categorical variables",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Dimension 1", fontsize=10)
    plt.ylabel("Dimension 2", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mca_categorielle.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # Final grades by higher education plans
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x="higher", y="G3", data=df, hue="higher", palette="Set2", legend=False
    )
    plt.title(
        "Final grades by higher education plans",
        fontsize=14,
    )
    plt.xlabel("Plans to pursue higher education", fontsize=10)
    plt.ylabel("Final grade (G3, out of 20)", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_Sup_G3.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # Mean final grades by school support and sex
    plt.figure(figsize=(12, 8))
    sns.barplot(x="schoolsup", y="G3", data=df, hue="sex", errorbar=None, palette="Pastel1")
    plt.title("Mean final grades by school support and sex", fontsize=14)
    plt.xlabel("Extra educational support", fontsize=10)
    plt.ylabel("Mean final grade (G3, out of 20)", fontsize=10)
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "barplot_Soutien_G3.png", dpi=300)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the exploratory figures.")
    parser.add_argument(
        "--show", action="store_true", help="Open the figures in plot windows."
    )
    args = parser.parse_args()
    plot_results(load_data(), show=args.show)
