# Déploiement Credit Risk — Streamlit + PySpark (sans FastAPI)

Architecture simplifiée :

```
R + sparklyr  →  model_credit/  →  Streamlit + PySpark (un seul service)
```

## Contenu du dossier `streamlit/`

```
streamlit/
├── app.py                # Application complète (UI + logique Spark)
├── model_credit/         # Pipeline Spark ML entraîné (ml_save() en R)
├── data/
│   └── credit_sample.csv # Dataset par défaut (400 lignes synthétiques)
├── requirements.txt       # Dépendances Python
├── packages.txt           # Dépendances système (Java, via apt)
├── runtime.txt             # Version Python pour Streamlit Cloud
└── .streamlit/config.toml # Thème et options serveur
```

## Déployer sur Streamlit Community Cloud

1. Poussez le dossier `streamlit/` (avec `model_credit/` et `data/`) dans un
   dépôt GitHub. C'est le **contenu de ce dossier** qui doit être à la racine
   du dépôt (ou indiquez `streamlit/app.py` comme "Main file path" si vous
   gardez la structure actuelle).
2. Sur [share.streamlit.io](https://share.streamlit.io), créez une nouvelle
   application en pointant vers `app.py`.
3. Streamlit Cloud lit automatiquement :
   - `requirements.txt` → installe Streamlit, pandas, plotly, pyspark ;
   - `packages.txt` → installe `openjdk-17-jdk-headless` via `apt-get`,
     indispensable pour que PySpark puisse démarrer une JVM ;
   - `runtime.txt` → fixe la version de Python.
4. Au premier chargement, l'application démarre une `SparkSession` locale
   (mise en cache), charge le modèle une seule fois, puis reste disponible
   pour toutes les prédictions suivantes.

Aucune variable d'environnement n'est requise : `JAVA_HOME` est détecté
automatiquement au démarrage (PATH, puis emplacements standards
`/usr/lib/jvm/...`). Vous pouvez vérifier le diagnostic (Java détecté,
JAVA_HOME, version Spark) dans l'onglet **Architecture** de l'application.

## Lancer en local

```bash
cd streamlit
pip install -r requirements.txt
# Java 17 doit être installé localement (apt install openjdk-17-jdk-headless
# sous Debian/Ubuntu, brew install openjdk@17 sous macOS, etc.)
streamlit run app.py
```

## Dataset par défaut vs dataset personnalisé

Dans l'onglet **Explorer un dataset**, un bouton radio permet de choisir :
- **Dataset par défaut** : `data/credit_sample.csv`, généré pour respecter
  le schéma attendu par le modèle (mêmes catégories que celles apprises
  par les `StringIndexer` du pipeline) ;
- **Charger mon propre CSV** : upload d'un fichier avec les colonnes
  `person_age, person_income, person_home_ownership, person_emp_length,
  loan_intent, loan_amnt, loan_int_rate, loan_percent_income,
  cb_person_default_on_file, cb_person_cred_hist_length`.

La prédiction batch (bouton "Lancer la prédiction batch") s'exécute avec
Spark directement dans le processus Streamlit, sans appel réseau à une API
externe.
