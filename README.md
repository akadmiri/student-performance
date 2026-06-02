# Student Performance Analysis
## Description
Ce projet implémente un pipeline modulaire et industrialisé de bout en bout pour le traitement, la visualisation statistique et la modélisation prédictive des performances académiques des étudiants en mathématiques. Basé sur le jeu de données réel d'UC Irvine (UCI Student Performance Dataset), le projet applique les principes de la Clean Architecture pour transformer des données brutes hétérogènes en insights actionnables et en prédictions robustes.

L'objectif métier s'inscrit dans une démarche analytique : segmenter les profils d'apprentissage et prédire la réussite à l'examen final ($G3$) afin de permettre des interventions pédagogiques ciblées.

---

## Architecture du Projet
Le projet est structuré de manière modulaire afin de séparer strictement les responsabilités (Data Processing, EDA, Modeling) :

```text
student-performance/
├── pyrightconfig.json      # Configuration de l'analyseur de types Pylance (pour ignorer les erreurs de typage)
├── requirements.txt        # Dépendances du projet
├── .gitignore              # Exclusion de l'environnement virtuel et des données
├── README.md               # Documentation du projet
├── data/
│   └── raw/                # Données brutes de l'UCI (Ignoré par Git)
├── output/                 # Graphiques et livrables d'analyse exportés
└── src/
    ├── preprocessing.py    # Pipeline de nettoyage, d'encodage et de mise à l'échelle
    ├── visualization.py    # Scripts d'analyse graphique (Distribution, ACM)
    └── analysis.py         # Entraînement de la régression linéaire et évaluation