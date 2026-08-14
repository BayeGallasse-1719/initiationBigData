import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType

# =============================================================================
# 1. CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Risque de Crédit | Spark ML",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. TAILWIND CSS CUSTOM
# =============================================================================
def apply_tailwind_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Main Header */
        .main-header {
            background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .main-title {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            color: white;
        }

        .sub-title {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1rem;
            font-weight: 400;
            margin-top: 0.5rem;
        }

        /* KPI Cards */
        .kpi-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border: 1px solid #E0F2FE;
        }

        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        }

        .kpi-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .kpi-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            background: #E0F2FE;
            color: #3B82F6;
        }

        .kpi-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
        }

        .kpi-num {
            font-size: 2rem;
            font-weight: 800;
            color: #1E40AF;
            line-height: 1.1;
        }

        .kpi-subtext {
            font-size: 0.8rem;
            color: #3B82F6;
            margin-top: 0.5rem;
            font-weight: 500;
        }

        /* Result Card */
        .res-card {
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }

        .res-card:hover {
            transform: translateY(-4px);
        }

        .res-high {
            background: linear-gradient(135deg, #E0F2FE 0%, #B3E5FC 100%);
            border: 2px solid #3B82F6;
            color: #1E40AF;
        }

        .res-low {
            background: linear-gradient(135deg, #E0F2FE 0%, #81D4FA 100%);
            border: 2px solid #3B82F6;
            color: #1E40AF;
        }

        .res-status {
            font-size: 1.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .res-percentage {
            font-size: 2.5rem;
            font-weight: 800;
            line-height: 1;
            margin: 0.8rem 0;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1E40AF !important;
            border-right: 1px solid #3B82F6;
        }

        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Primary Button */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #3B82F6 0%, #1E40AF 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 0.75rem 1.5rem !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
        }

        /* Input Fields */
        .stNumberInput, .stSelectbox, .stTextInput {
            border-radius: 8px !important;
            border: 1px solid #E0F2FE !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: #E0F2FE !important;
            border-radius: 8px !important;
            color: #1E40AF !important;
            font-weight: 600 !important;
        }

        /* Plotly Charts */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

apply_tailwind_css()

# =============================================================================
# 3. INITIALISATION DE SPARK ET CACHE
# =============================================================================
@st.cache_resource
def get_spark_session():
    """Initialise la session Spark locale."""
    return (SparkSession.builder
            .appName("StreamlitSparkCreditRisk")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .getOrCreate())

@st.cache_resource
def load_spark_model(model_path: str):
    """Charge le PipelineModel de Spark."""
    return PipelineModel.load(model_path)

@st.cache_data
def load_credit_data(csv_path: str = "../credit_risk_dataset.csv"):
    """Charge le dataset nettoyé pour les analyses du dashboard."""
    try:
        df = pd.read_csv(csv_path)
        df = df[df["person_age"] <= 100]
        df = df[df["person_emp_length"] <= 50]
        return df
    except Exception:
        return None

spark = get_spark_session()
df_raw = load_credit_data()

# =============================================================================
# 4. BARRE DE NAVIGATION (SIDEBAR)
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
            <div class="flex items-center gap-3 mb-4">
                <div class="bg-white text-blue-600 w-10 h-10 rounded-lg flex items-center justify-center text-xl">💳</div>
                <div>
                    <h3 class="text-white text-lg font-bold">Credit Analytics</h3>
                    <p class="text-blue-200 text-sm">Moteur PySpark ML</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=[
                "📊 Tableau de bord",
                "🔮 Formulaire de prédiction",
                "ℹ️ À propos"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        st.markdown("""
            <div class="bg-blue-100 text-blue-800 p-3 rounded-lg border border-blue-300">
                <div class="flex items-center gap-2 mb-1">
                    <span class="h-2 w-2 bg-green-500 rounded-full"></span>
                    <b>Spark Engine Actif</b>
                </div>
                <span class="text-blue-600 text-sm">Master: local[*]</span>
            </div>
        """, unsafe_allow_html=True)

        return page

page = render_sidebar()

# =============================================================================
# PAGE 1 : TABLEAU DE BORD DECISIONNEL
# =============================================================================
def render_dashboard():
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">Tableau de bord du portefeuille</h1>
            <p class="sub-title">Analyse décisionnelle de la distribution des risques et des prêts</p>
        </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.warning("⚠️ Fichier `credit_risk_dataset.csv` introuvable. Veuillez le placer dans le même répertoire que l'application.")
    else:
        # Filtres
        with st.expander("🛠️ Panneau de Filtrage Avancé", expanded=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                ownership_filter = st.multiselect(
                    "Statut de propriété",
                    options=sorted(df_raw["person_home_ownership"].dropna().unique()),
                    default=sorted(df_raw["person_home_ownership"].dropna().unique())
                )
            with f2:
                intent_filter = st.multiselect(
                    "Motif du prêt",
                    options=sorted(df_raw["loan_intent"].dropna().unique()),
                    default=sorted(df_raw["loan_intent"].dropna().unique())
                )
            with f3:
                age_min, age_max = int(df_raw["person_age"].min()), int(df_raw["person_age"].max())
                age_range = st.slider("Tranche d'âge du client", age_min, age_max, (age_min, age_max))

        # Filtrage du DataFrame
        filtered = df_raw[
            (df_raw["person_home_ownership"].isin(ownership_filter)) &
            (df_raw["loan_intent"].isin(intent_filter)) &
            (df_raw["person_age"].between(age_range[0], age_range[1]))
        ]

        # KPIs
        total = len(filtered)
        n_defaut = int(filtered["loan_status"].sum()) if total > 0 else 0
        taux_defaut = (n_defaut / total * 100) if total > 0 else 0
        revenu_moy = filtered["person_income"].mean() if total > 0 else 0
        pret_moy = filtered["loan_amnt"].mean() if total > 0 else 0

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Portefeuille</span>
                    <div class="kpi-icon">👥</div>
                </div>
                <div class="kpi-num">{total:,}</div>
                <div class="kpi-subtext">Demandes enregistrées</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Taux de Défaut</span>
                    <div class="kpi-icon">📉</div>
                </div>
                <div class="kpi-num">{taux_defaut:.1f}%</div>
                <div class="kpi-subtext">{n_defaut:,} clients à risque</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Revenu Moyen</span>
                    <div class="kpi-icon">💵</div>
                </div>
                <div class="kpi-num">{revenu_moy:,.0f} €</div>
                <div class="kpi-subtext">Par an par client</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-title">Prêt Moyen</span>
                    <div class="kpi-icon">🏦</div>
                </div>
                <div class="kpi-num">{pret_moy:,.0f} €</div>
                <div class="kpi-subtext">Exposition moyenne</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Graphiques Plotly
        plotly_template = {
            "layout": {
                "font": {"family": "Inter"},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            }
        }

        c1, c2 = st.columns(2)

        with c1:
            fig1 = px.histogram(
                filtered, x="loan_intent", color="loan_status",
                barmode="group",
                color_discrete_map={0: "#3B82F6", 1: "#1E40AF"},
                labels={"loan_intent": "Motif", "loan_status": "Statut", "count": "Effectif"},
                title="<b>Distribution des Défauts par Motif</b>",
            )
            fig1.update_layout(plotly_template["layout"])
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            sample = filtered.sample(min(2000, len(filtered)), random_state=42) if len(filtered) > 0 else filtered
            fig2 = px.scatter(
                sample, x="person_income", y="loan_amnt",
                color=sample["loan_status"].astype(str) if len(sample) > 0 else None,
                color_discrete_map={"0": "#3B82F6", "1": "#1E40AF"},
                opacity=0.7,
                labels={"person_income": "Revenu (€)", "loan_amnt": "Prêt (€)", "color": "Défaut"},
                title="<b>Revenu vs Montant du Prêt</b>",
            )
            fig2.update_layout(plotly_template["layout"])
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            fig3 = px.pie(
                filtered, names="person_home_ownership", hole=0.5,
                title="<b>Statut de Propriété Immobilière</b>",
                color_discrete_sequence=["#3B82F6", "#1E40AF", "#60A5FA", "#93C5FD"],
            )
            fig3.update_layout(plotly_template["layout"])
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            fig4 = px.box(
                filtered, x="loan_status", y="loan_int_rate",
                color="loan_status",
                color_discrete_map={0: "#3B82F6", 1: "#1E40AF"},
                labels={"loan_status": "Défaut (0=Non, 1=Oui)", "loan_int_rate": "Taux (%)"},
                title="<b>Impact du Taux d'Intérêt sur le Risque</b>",
            )
            fig4.update_layout(plotly_template["layout"])
            st.plotly_chart(fig4, use_container_width=True)

# =============================================================================
# PAGE 2 : FORMULAIRE D'INFERENCE (PREDICTION SPARK ML)
# =============================================================================
def render_prediction_form():
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">Évaluation du Risque Client</h1>
            <p class="sub-title">Passez un nouveau profil dans le pipeline d'inférence Spark ML</p>
        </div>
    """, unsafe_allow_html=True)

    col_gauche, col_droite = st.columns(2, gap="large")

    with col_gauche:
        st.markdown("### 👤 Informations du Demandeur")
        person_age = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Revenu annuel (€)", min_value=0, value=55000, step=1000)
        person_emp_length = st.number_input("Ancienneté professionnelle (années)", min_value=0.0, max_value=60.0, value=6.0, step=0.5)
        person_home_ownership = st.selectbox(
            "Statut de propriété",
            options=["RENT", "OWN", "MORTGAGE", "OTHER"],
            help="RENT = Locataire | OWN = Propriétaire | MORTGAGE = Hypothèque"
        )

    with col_droite:
        st.markdown("### 💳 Paramètres du Prêt")
        loan_intent = st.selectbox(
            "Motif de la demande",
            options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
        loan_amnt = st.number_input("Montant souhaité (€)", min_value=500, value=12000, step=500)
        loan_int_rate = st.number_input("Taux d'intérêt proposé (%)", min_value=0.0, max_value=40.0, value=10.5, step=0.1)

        st.markdown("### 📜 Historique Bancaire")
        cb_person_default_on_file = st.selectbox("Incidents de paiement enregistrés ?", options=["N", "Y"])
        cb_person_cred_hist_length = st.number_input("Ancienneté de l'historique de crédit (années)", min_value=0, max_value=50, value=5)

    st.markdown("---")

    # Schéma Spark ML
    schema = StructType([
        StructField("person_age", IntegerType(), True),
        StructField("person_income", DoubleType(), True),
        StructField("person_home_ownership", StringType(), True),
        StructField("person_emp_length", DoubleType(), True),
        StructField("loan_intent", StringType(), True),
        StructField("loan_amnt", DoubleType(), True),
        StructField("loan_int_rate", DoubleType(), True),
        StructField("cb_person_default_on_file", StringType(), True),
        StructField("cb_person_cred_hist_length", IntegerType(), True)
    ])

    input_data = [(
        int(person_age),
        float(person_income),
        person_home_ownership,
        float(person_emp_length),
        loan_intent,
        float(loan_amnt),
        float(loan_int_rate),
        cb_person_default_on_file,
        int(cb_person_cred_hist_length)
    )]

    model_dir = st.text_input("📁 Répertoire du modèle Spark ML sauvegardé", value="model_credit")

    if st.button("🚀 Calculer le Score de Risque Spark ML", type="primary", use_container_width=True):
        try:
            with st.spinner("Exécution de l'inférence via PySpark..."):
                df_spark = spark.createDataFrame(input_data, schema=schema)
                model = load_spark_model(f'../{model_dir}')
                predictions = model.transform(df_spark)

                row = predictions.select("prediction", "probability").collect()[0]
                pred_class = int(row["prediction"])
                prob_default = float(row["probability"].toArray()[1])

            st.markdown("---")
            st.markdown("## 📋 Décision d'Octroi du Prêt")

            r1, r2 = st.columns([1, 1], gap="medium")

            with r1:
                if pred_class == 1:
                    st.markdown(f"""
                    <div class="res-card res-high">
                        <div class="res-status">⚠️ RISQUE ÉLEVÉ DE DÉFAUT</div>
                        <p style="margin:0; font-size:0.9rem; opacity:0.8;">La prédiction indique un risque de non-remboursement élevé.</p>
                        <div class="res-percentage">{prob_default*100:.1f}%</div>
                        <span style="font-size:0.85rem; font-weight:600;">PROBABILITÉ DE DÉFAUT</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="res-card res-low">
                        <div class="res-status">✅ PROFIL ACCEPTÉ</div>
                        <p style="margin:0; font-size:0.9rem; opacity:0.8;">Le profil présente des garanties suffisantes pour l'accord du prêt.</p>
                        <div class="res-percentage">{prob_default*100:.1f}%</div>
                        <span style="font-size:0.85rem; font-weight:600;">PROBABILITÉ DE DÉFAUT</span>
                    </div>
                    """, unsafe_allow_html=True)

            with r2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_default * 100,
                    number={'suffix': "%", 'font': {'size': 36, 'family': 'Inter', 'color': '#1E40AF'}},
                    title={"text": "Indice de Risque de Crédit", "font": {"size": 16, "color": "#3B82F6"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#3B82F6"},
                        "bar": {"color": "#3B82F6", "thickness": 0.25},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 30], "color": "rgba(74, 222, 128, 0.2)"},
                            {"range": [30, 60], "color": "rgba(251, 191, 36, 0.2)"},
                            {"range": [60, 100], "color": "rgba(248, 113, 113, 0.2)"}
                        ],
                        "threshold": {"line": {"color": "#1E40AF", "width": 3}, "value": 50}
                    }
                ))
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter")
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 💡 Recommandation du Système")
            if prob_default < 0.30:
                st.success("🟢 **Validation Automatique** : Le profil respecte les critères de souscription.")
            elif prob_default < 0.50:
                st.warning("🟡 **Analyse Complémentaire** : Le niveau de risque est modéré. Des garanties complémentaires sont suggérées.")
            else:
                st.error("🔴 **Refus Recommandé** : L'exposition au risque dépasse le seuil toléré.")

        except Exception as e:
            st.error(f"❌ Erreur lors de l'exécution Spark : {e}")

# =============================================================================
# PAGE 3 : A PROPOS
# =============================================================================
def render_about():
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">À propos de l'application</h1>
            <p class="sub-title">Architecture Big Data et Modélisation Prédictive</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🏗️ Spécifications Techniques
    - **Engine Big Data** : PySpark (Session Spark en mode local `local[*]`)
    - **Algorithme ML** : Random Forest Classifier (`spark.ml`)
    - **Visualisation** : Plotly & Streamlit UI

    ### 📌 Fonctionnalités Clés
    - **Tableau de bord** : Analyse interactive des données de crédit avec des filtres dynamiques.
    - **Formulaire de prédiction** : Évaluation du risque client en temps réel via un modèle Spark ML.
    - **Design moderne** : Interface utilisateur intuitive et esthétique avec des cartes KPI et des graphiques avancés.

    ### 🔧 Technologies Utilisées
    - **Backend** : PySpark pour le traitement des données et l'apprentissage automatique.
    - **Frontend** : Streamlit pour l'interface utilisateur, Plotly pour les visualisations.
    - **Data** : Dataset de risque de crédit pour l'entraînement et l'évaluation du modèle.

    ### 📜 À Propos
    Cette application a été développée pour démontrer l'intégration de **PySpark ML** avec **Streamlit** afin de créer une solution complète pour l'analyse du risque de crédit.
    """)

# =============================================================================
# ROUTING DES PAGES
# =============================================================================
if page == "📊 Tableau de bord":
    render_dashboard()
elif page == "🔮 Formulaire de prédiction":
    render_prediction_form()
elif page == "ℹ️ À propos":
    render_about()