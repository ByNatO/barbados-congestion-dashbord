import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import base64
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

matplotlib.use('Agg')

from src.data_loader import load_full_data, get_test_data, filter_by_time
from src.model_loader import load_model
from src.prediction import predict_with_proba

# ------------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Congestion Barbade",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Styles CSS personnalisés (thème friendly)
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e3c72;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5b6f82;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9edf2 100%);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }

    .prediction-badge {
        display: block;
        margin: 1.5rem 0;
        padding: 1.2rem 2rem;
        border-radius: 20px;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        letter-spacing: 0.5px;
        transition: transform 0.2s;
    }
    .prediction-badge:hover {
        transform: scale(1.01);
    }
    .badge-free {
        background: linear-gradient(135deg, #11998e, #38ef7d);
    }
    .badge-light {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #333;
    }
    .badge-moderate {
        background: linear-gradient(135deg, #e65c00, #F9D423);
    }
    .badge-heavy {
        background: linear-gradient(135deg, #cb2d3e, #ef473a);
    }

    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.7rem 2.5rem;
        border-radius: 30px;
        border: none;
        box-shadow: 0 6px 15px rgba(30,60,114,0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        box-shadow: 0 8px 20px rgba(30,60,114,0.4);
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8faff 0%, #edf2f9 100%);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 0.8rem;
        border-left: 4px solid #2a5298;
        padding-left: 12px;
    }

    .banner-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    .banner-img {
        width: 100%;
        max-height: 200px;
        object-fit: cover;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Fonction pour convertir une image en base64
# ------------------------------------------------------------------
def get_base64_image(image_path):
    """Convertit une image en chaîne base64 pour affichage HTML."""
    try:
        with open(image_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode()
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{b64}"
    except FileNotFoundError:
        st.error(f"Image introuvable : {image_path}")
        return None

# ------------------------------------------------------------------
# Mapping jour de la semaine
# ------------------------------------------------------------------
DAY_MAPPING = {
    0.0: "Lundi",
    1.0: "Mardi",
    2.0: "Mercredi",
    3.0: "Jeudi",
    4.0: "Vendredi",
    5.0: "Samedi",
    6.0: "Dimanche"
}

# ------------------------------------------------------------------
# Entête principal
# ------------------------------------------------------------------
st.markdown('<div class="main-header">Gestion de la Congestion</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Un Rond - point de Barbade - Analyse prédictive du trafic</div>', unsafe_allow_html=True)

# Bandeau d'illustration (image en base64 pour une visibilité garantie)
banner_path = "assets/plan.png"
banner_b64 = get_base64_image(banner_path)
if banner_b64:
    st.markdown(
        f'<img src="{banner_b64}" class="banner-img" alt="Plan du rond-point">',
        unsafe_allow_html=True
    )
else:
    st.warning("Image d'illustration non disponible")

# ------------------------------------------------------------------
# Chargement des ressources (cache)
# ------------------------------------------------------------------
@st.cache_resource
def load_resources():
    model = load_model("models/mon_modele_xgb.pkl")
    full_df = load_full_data("data/barbadostraficcongestion_shifted_df.csv")
    test_df = get_test_data(full_df)
    return model, test_df

model, test_df = load_resources()

# ------------------------------------------------------------------
# Fonction utilitaire : arrondir l'heure à la valeur disponible la plus proche
# ------------------------------------------------------------------
def closest_hour(hour_float, available_hours):
    """Retourne l'heure la plus proche dans available_hours."""
    return min(available_hours, key=lambda x: abs(x - hour_float))

# ------------------------------------------------------------------
# Initialisation de l'état de session (fuseau horaire Paris)
# ------------------------------------------------------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'selected_day' not in st.session_state:
    # Obtenir le jour système à Paris
    now_paris = datetime.now(ZoneInfo("Europe/Paris"))
    system_day = float(now_paris.weekday())  # 0 = lundi
    st.session_state.selected_day = system_day
if 'selected_hour' not in st.session_state:
    now_paris = datetime.now(ZoneInfo("Europe/Paris"))
    system_hour = now_paris.hour + now_paris.minute / 60.0  # heure décimale
    available = sorted(test_df['hour'].unique())
    st.session_state.selected_hour = closest_hour(system_hour, available)
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

# ------------------------------------------------------------------
# Sidebar pour les contrôles (pré-remplie avec valeurs système)
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Paramètres de simulation")
    st.markdown("---")
    with st.expander("Description des niveaux de congestion"):
        st.markdown("""
        - **Free flowing** : Trafic fluide, aucune difficulté.
        - **Light delay** : Ralentissement léger.
        - **Moderate delay** : Congestion modérée, circulation ralentie.
        - **Heavy delay** : Congestion sévère, fortes perturbations.
        """)
    st.markdown("Choisissez le jour et l'heure pour évaluer la congestion.")
    st.markdown("---")
    day_options = sorted(test_df['day_week'].unique())
    selected_day = st.selectbox(
        "Jour de la semaine",
        options=day_options,
        index=day_options.index(st.session_state.selected_day) if st.session_state.selected_day in day_options else 0,
        format_func=lambda x: DAY_MAPPING.get(x, f"Jour {int(x)}"),
        key='day_selector'
    )
    hour_options = sorted(test_df['hour'].unique())
    selected_hour = st.selectbox(
        "Heure de la journée",
        options=hour_options,
        index=hour_options.index(st.session_state.selected_hour) if st.session_state.selected_hour in hour_options else 0,
        format_func=lambda x: f"{int(x)}h",
        key='hour_selector'
    )
    st.markdown("---")
    predict_btn = st.button("Analyser la congestion")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Développé par Omar Badiane</p>", unsafe_allow_html=True)

# Mettre à jour l'état avec les valeurs des widgets
st.session_state.selected_day = selected_day
st.session_state.selected_hour = selected_hour

# ------------------------------------------------------------------
# Fonction d'affichage des résultats pour un échantillon donné
# ------------------------------------------------------------------
def display_prediction(sample):
    pred_label, proba_dict = predict_with_proba(model, sample)

    # Badge coloré
    if "free" in pred_label.lower():
        badge_class = "badge-free"
    elif "light" in pred_label.lower():
        badge_class = "badge-light"
    elif "moderate" in pred_label.lower():
        badge_class = "badge-moderate"
    elif "heavy" in pred_label.lower():
        badge_class = "badge-heavy"
    else:
        badge_class = "badge-free"

    st.markdown(f'<div class="prediction-badge {badge_class}">Niveau : {pred_label.upper()}</div>', unsafe_allow_html=True)

    # Message contextuel
    if "heavy" in pred_label.lower():
        st.warning("Congestion sévère détectée. Il est recommandé d'envisager des mesures de régulation.")
    elif "moderate" in pred_label.lower():
        st.warning("Congestion modérée en cours. La situation peut évoluer.")
    elif "light" in pred_label.lower():
        st.info("Trafic légèrement ralenti. Restez attentif.")
    else:
        st.success("Trafic fluide. Aucune action particulière requise.")

    # Probabilités
    if proba_dict:
        st.markdown('<div class="section-title">Probabilités par niveau</div>', unsafe_allow_html=True)
        proba_df = pd.DataFrame({
            'Niveau': list(proba_dict.keys()),
            'Probabilité': list(proba_dict.values())
        }).sort_values('Probabilité', ascending=False)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = []
        for level in proba_df['Niveau']:
            if 'free' in level.lower():
                colors.append('#11998e')
            elif 'light' in level.lower():
                colors.append('#f7971e')
            elif 'moderate' in level.lower():
                colors.append('#e65c00')
            elif 'heavy' in level.lower():
                colors.append('#cb2d3e')
            else:
                colors.append('#95a5a6')
        bars = ax.barh(proba_df['Niveau'], proba_df['Probabilité'], color=colors, edgecolor='white', linewidth=1.2)
        ax.set_xlabel('Probabilité', fontweight='600')
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        for bar, p in zip(bars, proba_df['Probabilité']):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{p:.1%}', va='center', fontweight='600', fontsize=11)
        st.pyplot(fig)
    else:
        st.info("Probabilités non disponibles.")

    # Variables utilisées
    with st.expander("Voir les variables utilisées"):
        features = ['signaling', 'day_week', 'hour', 'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b']
        feat_df = pd.DataFrame({
            'Variable': features,
            'Valeur': [sample[f] for f in features]
        })
        st.table(feat_df)

    # Vidéo associée
    video_path = sample['video_path']
    with st.expander("Vidéo associée à cet échantillon"):
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.warning(f"Fichier vidéo introuvable : {video_path}")

    return pred_label

# ------------------------------------------------------------------
# Zone principale : résultats
# ------------------------------------------------------------------

# Déterminer si on doit exécuter une analyse (automatique au premier chargement ou sur clic)
run_analysis = False

# Premier lancement sans clic : on utilise les valeurs système
if not st.session_state.initialized:
    st.session_state.initialized = True
    run_analysis = True
    selected_day = st.session_state.selected_day
    selected_hour = st.session_state.selected_hour

if predict_btn:
    run_analysis = True

if run_analysis:
    sample = filter_by_time(test_df, selected_day, selected_hour)
    if sample is None:
        st.error("Aucun échantillon trouvé pour cette sélection.")
    else:
        st.markdown("---")
        st.markdown(f"### Résultat pour {DAY_MAPPING.get(selected_day, 'Jour')} à {int(selected_hour)}h")
        display_prediction(sample)

        # ---- Historique des heures précédentes ----
        st.markdown("---")
        st.markdown("### Historique des dernières heures")

        day_samples = test_df[test_df['day_week'] == selected_day]
        previous_hours = sorted(day_samples['hour'].unique())
        previous_hours = [h for h in previous_hours if h < selected_hour]
        if not previous_hours:
            st.info("Aucune heure précédente disponible pour ce jour.")
        else:
            previous_hours = previous_hours[-6:]
            history_data = []
            for h in reversed(previous_hours):
                row = filter_by_time(test_df, selected_day, h)
                if row is not None:
                    pred_label, proba_dict = predict_with_proba(model, row)
                    max_prob = max(proba_dict.values()) if proba_dict else 0.0
                    history_data.append({
                        'Heure': f"{int(h)}h",
                        'Niveau prédit': pred_label,
                        'Probabilité max': f"{max_prob:.1%}"
                    })
            if history_data:
                hist_df = pd.DataFrame(history_data)
                st.table(hist_df)

                st.markdown("#### Évolution du niveau de congestion")
                level_order = {"free flowing": 1, "light delay": 2, "moderate delay": 3, "heavy delay": 4}
                hist_df['Valeur'] = hist_df['Niveau prédit'].map(level_order)
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.plot(hist_df['Heure'], hist_df['Valeur'], marker='o', linestyle='-', color='#2a5298')
                ax2.set_yticks([1, 2, 3, 4])
                ax2.set_yticklabels(['Free', 'Light', 'Moderate', 'Heavy'])
                ax2.set_xlabel('Heure')
                ax2.set_ylabel('Niveau')
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            else:
                st.info("Aucune prédiction disponible pour les heures précédentes.")
else:
    st.info("Sélectionnez un jour et une heure dans le panneau latéral, puis cliquez sur **Analyser la congestion** pour obtenir une prédiction.")