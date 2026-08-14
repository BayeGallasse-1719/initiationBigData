import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import (
    StructType, StructField, DoubleType, StringType, IntegerType
)

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
# 2. DESIGN SYSTEM & CSS PERSONNALISÉ
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff !important;
    }
    
    .sub-title {
        color: #94a3b8 !important;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Cartes KPI Glassmorphism */
    .kpi-card-v2 {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem 1.25rem;
        border-radius: 16px;
        color: white;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card-v2:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.3);
    }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }

    .kpi-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }

    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }

    .kpi-num {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
    }

    .kpi-subtext {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* Boîte de résultat d'inférence */
    .res-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .res-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(185, 28, 28, 0.18) 100%);
        border: 2px solid #ef4444;
        color: #f87171;
    }
    
    .res-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(4, 120, 87, 0.18) 100%);
        border: 2px solid #10b981;
        color: #34d399;
    }

    .res-status {
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .res-percentage {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.8rem 0;
    }

    /* Custom Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }
    
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* Primary Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


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
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1rem;">
            <div style="background: #2563eb; width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">💳</div>
            <div>
                <h3 style="margin: 0; font-size: 1.1rem; font-weight: 700;">Credit Analytics</h3>
                <p style="margin: 0; font-size: 0.75rem; color: #94a3b8 !important;">Moteur PySpark ML</p>
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
        <div style="background: rgba(30, 41, 59, 0.5); padding: 0.8rem; border-radius: 10px; border: 1px solid #334155; font-size: 0.8rem;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <span style="height: 8px; width: 8px; background-color: #10b981; border-radius: 50%; display: inline-block;"></span>
                <b>Spark Engine Actif</b>
            </div>
            <span style="color: #94a3b8 !important;">Master: local[*]</span>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1 : TABLEAU DE BORD DECISIONNEL
# =============================================================================
if page == "📊 Tableau de bord":
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
            <div class="kpi-card-v2">
                <div class="kpi-header">
                    <span class="kpi-title">Portefeuille</span>
                    <div class="kpi-icon" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa;">👥</div>
                </div>
                <div class="kpi-num">{total:,}</div>
                <div class="kpi-subtext">Demandes enregistrées</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k2:
            st.markdown(f"""
            <div class="kpi-card-v2">
                <div class="kpi-header">
                    <span class="kpi-title">Taux de Défaut</span>
                    <div class="kpi-icon" style="background: rgba(239, 68, 68, 0.2); color: #f87171;">📉</div>
                </div>
                <div class="kpi-num">{taux_defaut:.1f}%</div>
                <div class="kpi-subtext" style="color: #f87171;">{n_defaut:,} clients à risque</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k3:
            st.markdown(f"""
            <div class="kpi-card-v2">
                <div class="kpi-header">
                    <span class="kpi-title">Revenu Moyen</span>
                    <div class="kpi-icon" style="background: rgba(16, 185, 129, 0.2); color: #34d399;">💵</div>
                </div>
                <div class="kpi-num">{revenu_moy:,.0f} €</div>
                <div class="kpi-subtext" style="color: #34d399;">Par an par client</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k4:
            st.markdown(f"""
            <div class="kpi-card-v2">
                <div class="kpi-header">
                    <span class="kpi-title">Prêt Moyen</span>
                    <div class="kpi-icon" style="background: rgba(168, 85, 247, 0.2); color: #c084fc;">🏦</div>
                </div>
                <div class="kpi-num">{pret_moy:,.0f} €</div>
                <div class="kpi-subtext" style="color: #c084fc;">Exposition moyenne</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Graphiques Plotly
        plotly_template = "plotly_dark"

        c1, c2 = st.columns(2)

        with c1:
            fig1 = px.histogram(
                filtered, x="loan_intent", color="loan_status",
                barmode="group",
                color_discrete_map={0: "#10b981", 1: "#ef4444"},
                labels={"loan_intent": "Motif", "loan_status": "Statut", "count": "Effectif"},
                title="<b>Distribution des Défauts par Motif</b>",
                template=plotly_template
            )
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            sample = filtered.sample(min(2000, len(filtered)), random_state=42) if len(filtered) > 0 else filtered
            fig2 = px.scatter(
                sample, x="person_income", y="loan_amnt",
                color=sample["loan_status"].astype(str) if len(sample) > 0 else None,
                color_discrete_map={"0": "#10b981", "1": "#ef4444"},
                opacity=0.7,
                labels={"person_income": "Revenu (€)", "loan_amnt": "Prêt (€)", "color": "Défaut"},
                title="<b>Revenu vs Montant du Prêt</b>",
                template=plotly_template
            )
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            fig3 = px.pie(
                filtered, names="person_home_ownership", hole=0.5,
                title="<b>Statut de Propriété Immobilière</b>",
                color_discrete_sequence=px.colors.qualitative.Bold,
                template=plotly_template
            )
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            fig4 = px.box(
                filtered, x="loan_status", y="loan_int_rate",
                color="loan_status",
                color_discrete_map={0: "#10b981", 1: "#ef4444"},
                labels={"loan_status": "Défaut (0=Non, 1=Oui)", "loan_int_rate": "Taux (%)"},
                title="<b>Impact du Taux d'Intérêt sur le Risque</b>",
                template=plotly_template
            )
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig4, use_container_width=True)


# =============================================================================
# PAGE 2 : FORMULAIRE D'INFERENCE (PREDICTION SPARK ML)
# =============================================================================
elif page == "🔮 Formulaire de prédiction":
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
                model = load_spark_model(f'{'../'}' + model_dir)
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
                        <div class="res-status">✅ PROFIL ACCEPTE (RISQUE FAIBLE)</div>
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
                    number={'suffix': "%", 'font': {'size': 36, 'family': 'Plus Jakarta Sans', 'color': '#ffffff'}},
                    title={"text": "Indice de Risque de Crédit", "font": {"size": 16, "color": "#94a3b8"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                        "bar": {"color": "#3b82f6", "thickness": 0.25},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 30], "color": "rgba(16, 185, 129, 0.2)"},
                            {"range": [30, 60], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [60, 100], "color": "rgba(239, 68, 68, 0.2)"}
                        ],
                        "threshold": {"line": {"color": "#ef4444", "width": 3}, "value": 50}
                    }
                ))
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Plus Jakarta Sans")
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
            st.error(f"Erreur lors de l'exécution Spark : {e}")


# =============================================================================
# PAGE 3 : A PROPOS
# =============================================================================
elif page == "ℹ️ À propos":
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">À propos de l'application</h1>
            <p class="sub-title">Architecture Big Data et Modélisation Predictive</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🏗️ Spécifications Techniques
    - **Engine Big Data** : PySpark (Session Spark en mode local `local[*]`)
    - **Algorithme ML** : Random Forest Classifier (`spark.ml`)
    - **Visualisation** : Plotly & Streamlit UI
    """)