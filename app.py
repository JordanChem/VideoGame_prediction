import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des ventes de jeux vidéo",
    page_icon="🎮",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c25 100%);
    }
    .stTitle {
        background: linear-gradient(120deg, #3494e6, #ec6ead);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        letter-spacing: 2px;
        font-size: 3em;
        margin-bottom: 1em;
    }
    .stSubheader {
        color: #e0e0e0 !important;
        text-align: center;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 2em 0;
    }
    .stButton>button {
        background: linear-gradient(120deg, #3494e6, #ec6ead);
        color: white;
        border: none;
        padding: 0.5em 2em;
        border-radius: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# Titre
st.markdown('<h1 class="stTitle">Analyse prédictive des ventes</h1>',unsafe_allow_html=True)

# Load data
with st.spinner('🔄 Chargement des données en cours...'):
    df = pd.read_excel('video_game_data.xlsx')
st.success('🎉 Données chargées avec succès!')

# Sidebar stylée
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1em; background: linear-gradient(120deg, rgba(52, 148, 230, 0.1), rgba(236, 106, 173, 0.1)); border-radius: 10px;'>
        <h2 style='color: #3494e6;'>🎮 Paramètres </h2>
    </div>
""", unsafe_allow_html=True)
release_date = st.sidebar.date_input("📅 Date de sortie prévue", value=datetime(2025, 12, 31))

# Paramètres d'entrée avec icônes
trailer_views = st.sidebar.number_input("🎥 Vues du trailer", min_value=0, value=100000, step=1000)
instagram_followers = st.sidebar.number_input("📸 Abonnés Instagram", min_value=0, value=50000, step=1000)
facebook_followers = st.sidebar.number_input("👥 Followers Facebook", min_value=0, value=75000, step=1000)
tiktok_followers = st.sidebar.number_input("🎵 Followers TikTok", min_value=0, value=25000, step=1000)

# Bouton de prédiction dans la sidebar
predict_button = st.sidebar.button("🚀 Prédire")

# Calculate lag_day for prediction
trailer_date = datetime(2025, 2, 12)
lag_day = (release_date - trailer_date.date()).days

# Aperçu des données avec style
with st.expander("📊 Aperçu des données d'entraînement", expanded=False):
    top_5_games = df.nlargest(5, 'global_sales')
    st.dataframe(top_5_games.style.format({'global_sales': '{:.2f} M'}))

# Préparation des modèles (code inchangé pour la logique)
youtube_model = LinearRegression()
youtube_features = df[['Total vues trailer', 'lag_day']].dropna()
youtube_target = df.loc[youtube_features.index, 'global_sales']
youtube_model.fit(youtube_features, youtube_target)

instagram_model = LinearRegression()
instagram_features = df[['Instagram']].dropna()
instagram_target = df.loc[instagram_features.index, 'global_sales']
instagram_model.fit(instagram_features, instagram_target)

facebook_model = LinearRegression()
facebook_features = df[['Facebook']].dropna()
facebook_target = df.loc[facebook_features.index, 'global_sales']
facebook_model.fit(facebook_features, facebook_target)

tiktok_model = LinearRegression()
tiktok_features = df[['Tiktok']].dropna()
tiktok_target = df.loc[tiktok_features.index, 'global_sales']
tiktok_model.fit(tiktok_features, tiktok_target)

if predict_button:
    # Prédictions
    youtube_pred = youtube_model.predict([[trailer_views, lag_day]])[0]
    instagram_pred = instagram_model.predict([[instagram_followers]])[0]
    facebook_pred = facebook_model.predict([[facebook_followers]])[0]
    tiktok_pred = tiktok_model.predict([[tiktok_followers]])[0]

    final_prediction = np.mean([youtube_pred, instagram_pred, facebook_pred, tiktok_pred])
    formatted_prediction = "{:,.0f}".format(final_prediction * 1000000).replace(",", " ")

    # Affichage stylé de la prédiction finale
    st.markdown("<h2 class='stSubheader'>🎯 Prévision des ventes</h2>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='background: linear-gradient(120deg, rgba(52, 148, 230, 0.1), rgba(236, 106, 173, 0.1));
                    padding: 2em; border-radius: 20px; margin: 2em 0; text-align: center;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.2);'>
            <h1 style='font-size: 4em; background: linear-gradient(120deg, #3494e6, #ec6ead);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                {formatted_prediction}
            </h1>
            <h3 style='color: #e0e0e0; margin: 0;'>unit&#233;s</h3>
        </div>
    """, unsafe_allow_html=True)

    # Section des graphiques avec style amélioré
    st.markdown("<h2 class='stSubheader'>📈 Analyse par canal de distribution</h2>", unsafe_allow_html=True)

    # Configuration des graphiques en grille
    col1, col2 = st.columns(2)

    # Style commun pour les graphiques
    graph_style = dict(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        title_font=dict(size=20),
        showlegend=True,
        margin=dict(t=50, l=50, r=50, b=50),
    )

    with col1:
        # YouTube views prediction plot
        views_range = np.linspace(df['Total vues trailer'].min(), df['Total vues trailer'].max(), 100)
        youtube_predictions = youtube_model.predict(np.column_stack((views_range, [lag_day]*100)))
        youtube_df = pd.DataFrame({'Vues YouTube': views_range, 'Ventes prédites': youtube_predictions})
        fig_youtube = px.line(youtube_df, x='Vues YouTube', y='Ventes prédites',
                             title='🎥 Impact des vues YouTube')
        fig_youtube.update_layout(**graph_style)
        st.plotly_chart(fig_youtube, use_container_width=True)

        # Facebook followers prediction plot
        facebook_range = np.linspace(df['Facebook'].min(), df['Facebook'].max(), 100)
        facebook_predictions = facebook_model.predict(facebook_range.reshape(-1, 1))
        facebook_df = pd.DataFrame({'Abonnés Facebook': facebook_range, 'Ventes prédites': facebook_predictions})
        fig_facebook = px.line(facebook_df, x='Abonnés Facebook', y='Ventes prédites',
                              title='👥 Impact de la présence Facebook')
        fig_facebook.update_layout(**graph_style)
        st.plotly_chart(fig_facebook, use_container_width=True)

    with col2:
        # Instagram followers prediction plot
        instagram_range = np.linspace(df['Instagram'].min(), df['Instagram'].max(), 100)
        instagram_predictions = instagram_model.predict(instagram_range.reshape(-1, 1))
        instagram_df = pd.DataFrame({'Abonnés Instagram': instagram_range, 'Ventes prédites': instagram_predictions})
        fig_instagram = px.line(instagram_df, x='Abonnés Instagram', y='Ventes prédites',
                               title='📸 Impact de la présence Instagram')
        fig_instagram.update_layout(**graph_style)
        st.plotly_chart(fig_instagram, use_container_width=True)

        # TikTok followers prediction plot
        tiktok_range = np.linspace(df['Tiktok'].min(), df['Tiktok'].max(), 100)
        tiktok_predictions = tiktok_model.predict(tiktok_range.reshape(-1, 1))
        tiktok_df = pd.DataFrame({'Abonnés TikTok': tiktok_range, 'Ventes prédites': tiktok_predictions})
        fig_tiktok = px.line(tiktok_df, x='Abonnés TikTok', y='Ventes prédites',
                             title='🎵 Impact de la présence TikTok')
        fig_tiktok.update_layout(**graph_style)
        st.plotly_chart(fig_tiktok, use_container_width=True)
