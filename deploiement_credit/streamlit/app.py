
import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# 0. Configuration Java / PySpark — DOIT s'exécuter avant tout import pyspark
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(APP_DIR / "model_credit"))
DEFAULT_DATA_PATH = APP_DIR / "data" / "credit_sample.csv"


def _configure_java_home() -> str | None:
    """Trouve un JDK utilisable (Streamlit Cloud installe openjdk via packages.txt)
    et positionne JAVA_HOME si besoin, pour que PySpark démarre correctement."""
    if os.getenv("JAVA_HOME") and Path(os.environ["JAVA_HOME"]).exists():
        return os.environ["JAVA_HOME"]

    # 1) java déjà sur le PATH (cas le plus courant sur Streamlit Cloud + packages.txt)
    java_bin = shutil.which("java")
    if java_bin:
        try:
            real_bin = Path(os.path.realpath(java_bin))
            # .../<JAVA_HOME>/bin/java -> on remonte de 2 niveaux
            candidate = real_bin.parent.parent
            if (candidate / "bin" / "java").exists():
                os.environ["JAVA_HOME"] = str(candidate)
                return str(candidate)
        except Exception:
            pass

    # 2) emplacements Debian/Ubuntu classiques (image Streamlit Cloud)
    for pattern in ("/usr/lib/jvm/java-17-openjdk*", "/usr/lib/jvm/java-11-openjdk*",
                     "/usr/lib/jvm/default-java", "/usr/lib/jvm/java-21-openjdk*"):
        import glob
        for path in glob.glob(pattern):
            if Path(path, "bin", "java").exists():
                os.environ["JAVA_HOME"] = path
                return path

    return os.environ.get("JAVA_HOME")


_configure_java_home()
os.environ.setdefault("PYSPARK_PYTHON", "python3")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python3")

# Import pyspark seulement après avoir réglé JAVA_HOME
from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    DoubleType, StringType, StructField, StructType,
)

REQUIRED_COLUMNS = [
    "person_age", "person_income", "person_home_ownership",
    "person_emp_length", "loan_intent", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]
NUMERIC_COLUMNS = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CATEGORICAL_COLUMNS = [
    "person_home_ownership", "loan_intent", "cb_person_default_on_file",
]

SPARK_SCHEMA = StructType([
    StructField("person_age", DoubleType(), True),
    StructField("person_income", DoubleType(), True),
    StructField("person_home_ownership", StringType(), True),
    StructField("person_emp_length", DoubleType(), True),
    StructField("loan_intent", StringType(), True),
    StructField("loan_amnt", DoubleType(), True),
    StructField("loan_int_rate", DoubleType(), True),
    StructField("loan_percent_income", DoubleType(), True),
    StructField("cb_person_default_on_file", StringType(), True),
    StructField("cb_person_cred_hist_length", DoubleType(), True),
])

st.set_page_config(
    page_title="Credit Risk • Spark ML",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 1. Ressources mises en cache : session Spark + modèle
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Démarrage de Spark et chargement du modèle...")
def get_spark_and_model():
    java_home = os.environ.get("JAVA_HOME", "non détecté")
    spark = (
        SparkSession.builder
        .appName("CreditRiskStreamlit")
        .master("local[*]")
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g"))
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(MODEL_PATH)
    return spark, model, java_home


def get_model_labels(model) -> dict[str, list[str]]:
    labels = {}
    for stage in model.stages:
        if stage.__class__.__name__ == "StringIndexerModel":
            col = stage.getInputCol()
            labels[col] = [str(x) for x in stage.labels]
    return labels


try:
    spark, model, java_home_used = get_spark_and_model()
    MODEL_LABELS = get_model_labels(model)
    SPARK_OK = True
    SPARK_ERROR = None
except Exception as exc:  # noqa: BLE001
    SPARK_OK = False
    SPARK_ERROR = str(exc)
    MODEL_LABELS = {}
    java_home_used = os.environ.get("JAVA_HOME", "non détecté")


# ---------------------------------------------------------------------------
# 2. Fonctions métier (validation, préparation, prédiction) — ex-FastAPI
# ---------------------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame) -> dict:
    errors, warnings = [], []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        errors.append({"type": "colonnes_manquantes", "columns": missing})
        return {
            "compatible": False, "rows": int(len(df)), "columns": list(df.columns),
            "missing_columns": missing, "errors": errors, "warnings": warnings,
        }

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS and c != "loan_status"]
    if extra:
        warnings.append({"type": "colonnes_ignorées", "columns": extra})

    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = int(((df[col].notna()) & (converted.isna())).sum())
        if bad:
            errors.append({"type": "type_incompatible", "column": col,
                            "message": f"{bad} valeur(s) non numériques."})

    for col in CATEGORICAL_COLUMNS:
        allowed = set(MODEL_LABELS.get(col, []))
        if allowed:
            values = set(df[col].dropna().astype(str).unique())
            unknown = sorted(values - allowed)
            if unknown:
                errors.append({
                    "type": "categorie_inconnue", "column": col,
                    "unknown_values": unknown, "allowed_values": sorted(allowed),
                })

    return {
        "compatible": len(errors) == 0, "rows": int(len(df)), "columns": list(df.columns),
        "missing_columns": missing, "errors": errors, "warnings": warnings,
    }


def prepare_spark_df(df: pd.DataFrame):
    work = df.copy()
    for col in NUMERIC_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("float64")
    for col in CATEGORICAL_COLUMNS:
        work[col] = work[col].astype(object).where(work[col].notna(), None)
    return spark.createDataFrame(work[REQUIRED_COLUMNS], schema=SPARK_SCHEMA)


def run_batch_prediction(df: pd.DataFrame) -> dict:
    spark_df = prepare_spark_df(df)
    prediction_df = model.transform(spark_df)
    rows = prediction_df.select("prediction", "probability").collect()

    preds, probs = [], []
    for row in rows:
        prob = row["probability"]
        p1 = float(prob[1]) if prob is not None and len(prob) > 1 else None
        preds.append(int(row["prediction"]))
        probs.append(p1)

    out = df.copy()
    out["prediction"] = preds
    out["probability_default"] = probs
    return {
        "rows": int(len(out)),
        "default_count": int(out["prediction"].sum()),
        "default_rate": float(out["prediction"].mean()) if len(out) else 0.0,
        "full_results": out,
    }


def predict_individual(payload: dict) -> dict:
    df = pd.DataFrame([payload])
    spark_df = prepare_spark_df(df)
    pred_df = model.transform(spark_df)
    result = pred_df.select("prediction", "probability").collect()[0]

    proba = float(result["probability"][1])
    pred = int(result["prediction"])

    if proba < 0.30:
        risk, msg = "Faible", "Risque de défaut faible."
    elif proba < 0.60:
        risk, msg = "Modéré", "Risque de défaut modéré – vigilance recommandée."
    else:
        risk, msg = "Élevé", "Risque de défaut élevé – attention particulière requise."

    return {"prediction": pred, "probability_default": round(proba, 4),
            "risk_level": risk, "message": msg}


def create_gauge(probability: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 42, "color": "#0b2f6b"}},
        title={"text": "Probabilité de défaut", "font": {"size": 18, "color": "#334155"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#1769e0"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 30], "color": "#d1fae5"},
                {"range": [30, 60], "color": "#fef3c7"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "#dc2626", "width": 4}, "thickness": 0.8,
                          "value": probability * 100},
        },
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=280,
                       paper_bgcolor="rgba(0,0,0,0)", font={"color": "#0f172a"})
    return fig


# ---------------------------------------------------------------------------
# 3. CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background: #f4f7fc; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b2f6b 0%, #1256b5 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
.hero {
    background: linear-gradient(135deg, #0b2f6b 0%, #1769e0 100%);
    padding: 28px 32px; border-radius: 20px; color: white; margin-bottom: 24px;
    box-shadow: 0 12px 30px rgba(23,105,224,.18);
}
.hero h1 { margin: 0; font-size: 2.1rem; }
.hero p { margin: 8px 0 0; opacity: .9; }
.card {
    background: white; border: 1px solid #e4edfb; border-radius: 16px;
    padding: 20px; box-shadow: 0 8px 24px rgba(22,64,120,.07); margin-bottom: 16px;
}
.metric-title { color: #6680a8; font-size: .8rem; font-weight: 700; text-transform: uppercase; }
.metric-value { color: #0b2f6b; font-size: 1.7rem; font-weight: 800; margin-top: 4px; }
.ok  { background:#ecfdf5; border:1px solid #a7f3d0; color:#047857; border-radius:12px; padding:12px 16px; font-weight:700; }
.bad { background:#fff7ed; border:1px solid #fed7aa; color:#c2410c; border-radius:12px; padding:12px 16px; font-weight:700; }
.small-note { color:#6b7f9e; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 4. Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💳 Credit Risk")
    st.caption("Streamlit + PySpark (tout-en-un)")
    st.divider()

    if SPARK_OK:
        st.success("Spark & modèle chargés")
        st.caption(f"Spark : {spark.version}")
        st.caption(f"JAVA_HOME : {java_home_used}")
    else:
        st.error("Échec du démarrage de Spark")
        st.caption("Voir la page Architecture pour le diagnostic.")

    page = st.radio(
        "Navigation",
        ["📊 Tableau de bord", "📁 Explorer un dataset", "🎯 Prédiction individuelle", "ℹ️ Architecture"],
    )

# ---------------------------------------------------------------------------
# 5. Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Analyse du risque de crédit</h1>
    <p>Pipeline Spark ML (R/sparklyr) intégré directement à Streamlit – aucune API externe</p>
</div>
""", unsafe_allow_html=True)

if not SPARK_OK:
    st.error(
        "Spark n'a pas pu démarrer. Vérifiez que Java est installé "
        "(fichier `packages.txt` avec `openjdk-17-jdk-headless` sur Streamlit Cloud)."
    )
    with st.expander("Détails de l'erreur"):
        st.code(SPARK_ERROR or "Erreur inconnue")
    st.stop()

# ---------------------------------------------------------------------------
# 6. PAGE — Tableau de bord
# ---------------------------------------------------------------------------
if page == "📊 Tableau de bord":
    st.subheader("Vue d'ensemble")
    c1, c2, c3, c4 = st.columns(4)
    for col, title, value in [
        (c1, "Moteur", "Spark ML"),
        (c2, "Backend", "Intégré (Streamlit)"),
        (c3, "Frontend", "Streamlit"),
        (c4, "Version Spark", spark.version),
    ]:
        with col:
            st.markdown(
                f'<div class="card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("""
    <div class="card">
        <b>Workflow recommandé</b><br><br>
        1. <b>Explorer un dataset</b> → utilisez le jeu de données par défaut ou chargez un CSV<br>
        2. <b>Prédiction individuelle</b> → formulaire pour un client unique + jauge de risque<br>
        3. Les prédictions batch se lancent directement depuis la page d'exploration
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 7. PAGE — Explorer un dataset
# ---------------------------------------------------------------------------
elif page == "📁 Explorer un dataset":
    st.subheader("Explorer un jeu de données")
    st.markdown(
        '<p class="small-note">Utilisez le jeu de données par défaut fourni avec l\'application, '
        'ou chargez votre propre CSV.</p>',
        unsafe_allow_html=True,
    )

    source = st.radio(
        "Source des données",
        ["📦 Dataset par défaut", "⬆️ Charger mon propre CSV"],
        horizontal=True,
    )

    df = None
    if source == "📦 Dataset par défaut":
        if DEFAULT_DATA_PATH.exists():
            df = pd.read_csv(DEFAULT_DATA_PATH)
            st.success(f"Dataset par défaut chargé : **{len(df):,}** lignes • **{len(df.columns)}** colonnes")
        else:
            st.error("Le fichier de données par défaut est introuvable (data/credit_sample.csv).")
    else:
        uploaded = st.file_uploader("Choisir un fichier CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(BytesIO(uploaded.getvalue()))
                st.success(f"Fichier chargé : **{len(df):,}** lignes • **{len(df.columns)}** colonnes")
            except Exception as e:
                st.error(f"Impossible de lire le fichier : {e}")

    if df is not None:
        st.markdown("### Aperçu (5 premières lignes)")
        st.dataframe(df.head(5), use_container_width=True, height=260)

        # ---------- Filtres ----------
        st.markdown("### Filtres rapides")
        filter_cols = st.columns(4)

        home_options = ["Tous"] + sorted(df["person_home_ownership"].dropna().astype(str).unique().tolist()) if "person_home_ownership" in df.columns else ["Tous"]
        with filter_cols[0]:
            home_filter = st.selectbox("Home Ownership", home_options)

        intent_options = ["Tous"] + sorted(df["loan_intent"].dropna().astype(str).unique().tolist()) if "loan_intent" in df.columns else ["Tous"]
        with filter_cols[1]:
            intent_filter = st.selectbox("Loan Intent", intent_options)

        default_options = ["Tous"] + sorted(df["cb_person_default_on_file"].dropna().astype(str).unique().tolist()) if "cb_person_default_on_file" in df.columns else ["Tous"]
        with filter_cols[2]:
            default_filter = st.selectbox("Default on file", default_options)

        with filter_cols[3]:
            if "person_age" in df.columns:
                age_min, age_max = int(df["person_age"].min()), int(df["person_age"].max())
                age_range = st.slider("Âge", age_min, age_max, (age_min, age_max))
            else:
                age_range = None

        filtered = df.copy()
        if home_filter != "Tous" and "person_home_ownership" in filtered.columns:
            filtered = filtered[filtered["person_home_ownership"].astype(str) == home_filter]
        if intent_filter != "Tous" and "loan_intent" in filtered.columns:
            filtered = filtered[filtered["loan_intent"].astype(str) == intent_filter]
        if default_filter != "Tous" and "cb_person_default_on_file" in filtered.columns:
            filtered = filtered[filtered["cb_person_default_on_file"].astype(str) == default_filter]
        if age_range and "person_age" in filtered.columns:
            filtered = filtered[(filtered["person_age"] >= age_range[0]) & (filtered["person_age"] <= age_range[1])]

        st.info(f"**{len(filtered):,}** lignes après filtrage")

        st.markdown("### Visualisations")
        g1, g2 = st.columns(2)
        with g1:
            if "loan_intent" in filtered.columns:
                fig1 = px.histogram(filtered, x="loan_intent", title="Répartition par Loan Intent",
                                     color_discrete_sequence=["#1769e0"])
                fig1.update_layout(margin=dict(t=50, b=20), height=350)
                st.plotly_chart(fig1, use_container_width=True)
        with g2:
            if "person_home_ownership" in filtered.columns:
                fig2 = px.pie(filtered, names="person_home_ownership", title="Home Ownership",
                              color_discrete_sequence=px.colors.qualitative.Set2)
                fig2.update_layout(margin=dict(t=50, b=20), height=350)
                st.plotly_chart(fig2, use_container_width=True)

        if "person_income" in filtered.columns and "loan_amnt" in filtered.columns:
            fig3 = px.scatter(filtered, x="person_income", y="loan_amnt",
                               color="loan_intent" if "loan_intent" in filtered.columns else None,
                               title="Revenu vs Montant du prêt", opacity=0.6)
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)

        # ---------- Prédiction batch (in-process, sans API) ----------
        st.markdown("---")
        if st.button("🚀 Lancer la prédiction batch (Spark, local)", type="primary", use_container_width=True):
            validation = validate_dataframe(filtered)
            if not validation["compatible"]:
                st.error("Le jeu de données n'est pas compatible avec le modèle.")
                st.json(validation["errors"])
            else:
                with st.spinner("Prédiction en cours via Spark..."):
                    try:
                        result = run_batch_prediction(filtered)
                        st.success(f"Terminé – {result['rows']:,} lignes analysées")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Défauts prédits", result["default_count"])
                        m2.metric("Taux de défaut", f"{result['default_rate']:.2%}")
                        m3.metric("Lignes", f"{result['rows']:,}")
                        st.dataframe(result["full_results"].head(50), use_container_width=True)

                        csv_bytes = result["full_results"].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Télécharger les résultats complets (CSV)",
                            data=csv_bytes,
                            file_name="predictions_credit_risk.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Erreur de prédiction : {exc}")

# ---------------------------------------------------------------------------
# 8. PAGE — Prédiction individuelle
# ---------------------------------------------------------------------------
elif page == "🎯 Prédiction individuelle":
    st.subheader("Prédiction pour un individu")
    st.markdown(
        '<p class="small-note">Remplissez le formulaire. Le modèle Spark (chargé dans ce même '
        'processus Streamlit) retourne la probabilité de défaut et une jauge visuelle.</p>',
        unsafe_allow_html=True,
    )

    labels = MODEL_LABELS

    with st.form("individual_form"):
        col1, col2 = st.columns(2)

        with col1:
            person_age = st.number_input("Âge", min_value=18, max_value=100, value=35)
            person_income = st.number_input("Revenu annuel ($)", min_value=0, value=65000, step=1000)
            person_emp_length = st.number_input("Ancienneté emploi (années)", min_value=0.0, value=5.0, step=0.5)
            loan_amnt = st.number_input("Montant du prêt ($)", min_value=0, value=12000, step=500)
            loan_int_rate = st.number_input("Taux d'intérêt (%)", min_value=0.0, value=11.5, step=0.1)

        with col2:
            home_opts = labels.get("person_home_ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            person_home_ownership = st.selectbox("Type de logement", home_opts)

            intent_opts = labels.get("loan_intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            loan_intent = st.selectbox("Objet du prêt", intent_opts)

            default_opts = labels.get("cb_person_default_on_file", ["Y", "N"])
            cb_person_default_on_file = st.selectbox("Défaut antérieur (fichier crédit)", default_opts)

            loan_percent_income = st.number_input("Ratio prêt / revenu", min_value=0.0, max_value=1.0, value=0.18, step=0.01)
            cb_person_cred_hist_length = st.number_input("Longueur historique crédit (années)", min_value=0, value=8)

        submitted = st.form_submit_button("Obtenir la prédiction", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "person_age": person_age, "person_income": person_income,
            "person_home_ownership": person_home_ownership, "person_emp_length": person_emp_length,
            "loan_intent": loan_intent, "loan_amnt": loan_amnt, "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }

        with st.spinner("Calcul du modèle Spark..."):
            try:
                result = predict_individual(payload)
                proba = result["probability_default"]
                risk = result["risk_level"]

                st.markdown("---")
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    st.plotly_chart(create_gauge(proba), use_container_width=True)
                with c2:
                    st.markdown("### Résultat")
                    if risk == "Faible":
                        st.markdown(f'<div class="ok">Risque : <b>{risk}</b></div>', unsafe_allow_html=True)
                    elif risk == "Modéré":
                        st.warning(f"Risque : **{risk}**")
                    else:
                        st.markdown(f'<div class="bad">Risque : <b>{risk}</b></div>', unsafe_allow_html=True)

                    st.metric("Probabilité de défaut", f"{proba:.2%}")
                    st.metric("Classe prédite", "Défaut" if result["prediction"] == 1 else "Non-défaut")
                    st.info(result["message"])
            except Exception as e:  # noqa: BLE001
                st.error(f"Erreur de prédiction : {e}")

# ---------------------------------------------------------------------------
# 9. PAGE — Architecture
# ---------------------------------------------------------------------------
else:
    st.subheader("Architecture")
    st.markdown("""
    <div class="card">
        <b>1. Entraînement (R + sparklyr)</b><br>
        Pipeline sauvegardé avec <code>ml_save()</code> dans le dossier <code>model_credit/</code>.<br><br>
        <b>2. Application (Streamlit + PySpark, tout-en-un)</b><br>
        - Une <code>SparkSession</code> locale est démarrée directement dans le processus Streamlit
          (mise en cache via <code>st.cache_resource</code>, un seul démarrage par instance).<br>
        - Le modèle Spark ML est chargé une seule fois et réutilisé pour toutes les prédictions.<br>
        - Plus aucune API FastAPI séparée : tout tourne dans le même service.<br><br>
        <b>3. Java sur Streamlit Cloud</b><br>
        - PySpark a besoin d'un JDK. Le fichier <code>packages.txt</code> à la racine du dossier
          déployé demande à Streamlit Cloud d'installer <code>openjdk-17-jdk-headless</code> via apt.<br>
        - Au démarrage, l'application détecte automatiquement <code>JAVA_HOME</code> (variable
          d'environnement, PATH, ou emplacements standards Debian/Ubuntu).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Diagnostic environnement")
    diag_cols = st.columns(3)
    with diag_cols[0]:
        st.metric("Java détecté", "Oui" if shutil.which("java") else "Non")
    with diag_cols[1]:
        st.metric("JAVA_HOME", os.environ.get("JAVA_HOME", "—"))
    with diag_cols[2]:
        st.metric("Spark", spark.version if SPARK_OK else "Indisponible")

    with st.expander("Sortie de `java -version`"):
        try:
            out = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10)
            st.code((out.stderr or out.stdout or "Aucune sortie").strip())
        except Exception as e:  # noqa: BLE001
            st.code(f"Impossible d'exécuter java -version : {e}")
