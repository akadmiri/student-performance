# Student Performance Analysis Pipeline

## Description

Ce projet implémente un pipeline modulaire pour le traitement, la visualisation et la modélisation prédictive des performances académiques d'étudiants en mathématiques. Basé sur l'[UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance), il transforme des données brutes hétérogènes en insights actionnables et en prédictions robustes.

L'objectif s'inscrit dans une démarche de *Learning Analytics* : segmenter les profils d'apprentissage et prédire la réussite à l'examen final ($G3$) afin de permettre des interventions pédagogiques ciblées.

---

## Architecture du Projet

```text
student-performance/
├── pyrightconfig.json      # Configuration du type checker Pylance (pour ignorer des errurs te typage)
├── requirements.txt        # Dépendances du projet
├── .gitignore
├── README.md
├── data/
│   └── raw/                # Données brutes UCI (ignoré par Git)
├── output/                 # Graphiques exportés 
└── src/
    ├── preprocessing.py    # Nettoyage, encodage et mise à l'échelle
    ├── visualization.py    # Analyse graphique (distributions, ACM, boxplots)
    └── analysis.py         # Régression linéaire et évaluation des performances
```

---

## Pipeline de Preprocessing

Pour éviter toute fuite de données (*data leakage*), le preprocessing suit un protocole strict :

1. **Partitionnement préalable :** séparation entraînement/test ($80\%$/$20\%$) avant toute transformation.
2. **Branche numérique :** standardisation via `StandardScaler` ($\mu = 0,\ \sigma = 1$) pour garantir la stabilité numérique du calcul matriciel.
3. **Branche catégorielle :** encodage one-hot via `OneHotEncoder(handle_unknown='ignore')`, faisant passer la dimension de la matrice de 32 à 58 variables.

L'orchestration est assurée par un `ColumnTransformer` Scikit-Learn appliqué exclusivement sur les données d'entraînement, puis transformé sur le jeu de test.

---

## Résultats et Insights

### Distribution de la variable cible ($G3$)

La distribution des notes finales révèle une concentration anormale d'individus à $0/20$. Ce phénomène traduit un décrochage en cours d'année plutôt qu'un niveau académique nul — une distribution censurée à interpréter avec précaution dans la modélisation.

### Analyse des Correspondances Multiples (ACM)

L'ACM appliquée à l'espace qualitatif fait apparaître un regroupement géométrique net entre les facteurs d'isolement (absence d'accès Internet, zone rurale, parent inactif) et le décrochage scolaire, fournissant un levier de ciblage prédictif exploitable en amont des examens.

### Performances du Modèle

Régression linéaire multiple par Moindres Carrés Ordinaires (MCO), évaluée sur le jeu de test indépendant :

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **R²** | `0.7859` | 78,6 % de la variance de $G3$ expliquée par le modèle |
| **RMSE** | `2.1936` | Écart moyen de ±2,19 points entre note prédite et réelle |

**Note méthodologique :** ces performances sont soutenues par la forte colinéarité des notes intermédiaires $G1$ et $G2$ avec $G3$. Un modèle alternatif excluant ces variables — basé uniquement sur les facteurs socio-comportementaux — serait plus pertinent en production pour détecter le risque de décrochage dès le début de l'année scolaire, avant tout examen.

---

## Installation et Utilisation

```bash
# Cloner le dépôt
git clone https://github.com/akadmiri/student-performance.git
cd student-performance

# Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

Télécharger le dataset depuis [UCI](https://archive.ics.uci.edu/dataset/320/student+performance) et placer `student-mat.csv` dans `data/raw/`.

```bash
# Preprocessing
python3 src/preprocessing.py

# Visualisations (exportées dans output/)
python3 src/visualization.py

# Entraînement et évaluation du modèle
python3 src/analysis.py
```

---

## Stack Technique

`Python 3.11` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn` · `prince`
