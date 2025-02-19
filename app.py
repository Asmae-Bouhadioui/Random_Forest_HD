import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib

# Configuration de la page en mode large
st.set_page_config(page_title="Dashboard Médical", layout="wide")

# CSS pour supprimer les marges
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 95%;
        }
    </style>
""", unsafe_allow_html=True)

# Charger le modèle entraîné
model = joblib.load('RF_Model.pkl')

# Interface de chargement de fichier
st.sidebar.header("Charger votre dataset")
uploaded_file = st.sidebar.file_uploader("Choisir un fichier CSV", type=["csv"])

# Si un fichier est téléchargé
if uploaded_file is not None:
    # Charger le dataset
    data = pd.read_csv(uploaded_file)
    # Aperçu des données avec meilleure visibilité
    st.title("📌 Aperçu des données")
    st.dataframe(data.head())  # Afficher les premières lignes du dataset


    # ---- DASHBOARD ----
    st.write("## 📊 Dashboard d'Analyse des Données")

    # Utilisation de st.container() pour organiser l'affichage
    with st.container():
        col1, col2 = st.columns(2)

        # 📌 1. Distribution de l'Âge avec une ligne d'évolution
        with col1:
            st.subheader("1️⃣ Distribution de l'Âge")
            fig_age = px.histogram(data, x='age', nbins=20, title='Distribution de l\'Âge', labels={'age': 'Âge'}, opacity=0.7)
            age_counts = data['age'].value_counts().sort_index()
            fig_age.add_trace(go.Scatter(x=age_counts.index, y=age_counts.values, mode='lines+markers', name='Tendance'))
            st.plotly_chart(fig_age, use_container_width=True)

        # 📌 2. Relation entre l'Âge et la Fréquence Cardiaque Maximale
        with col2:
            st.subheader("2️⃣ Âge vs Fréquence Cardiaque Max")
            fig_thalach = px.scatter(data, x='age', y='thalach', color='target', title="Âge vs. Fréquence Cardiaque Max")
            st.plotly_chart(fig_thalach, use_container_width=True)

    with st.container():
        col3, col4 = st.columns(2)

        # 📌 3. Présence de la Maladie Cardiaque selon le Sexe
        with col3:
            st.subheader("3️⃣ Maladie Cardiaque par Sexe")
            fig_sex = px.histogram(data, x='sex', color='target', title='Présence de la Maladie Cardiaque selon le Sexe', labels={'sex': 'Sexe'})
            st.plotly_chart(fig_sex, use_container_width=True)

        # 📌 4. Type de Douleur Thoracique vs Maladie Cardiaque
        with col4:
            st.subheader("4️⃣ Douleur Thoracique vs Maladie Cardiaque")
            fig_cp = px.histogram(data, x='cp', color='target', title='Type de Douleur Thoracique vs Maladie Cardiaque')
            st.plotly_chart(fig_cp, use_container_width=True)

    # 📌 5. Matrice de Corrélation (plein écran)
    st.subheader("5️⃣ Matrice de Corrélation")
    corr_matrix = data.corr()
    fig_corr = px.imshow(corr_matrix, title="Matrice de Corrélation")
    st.plotly_chart(fig_corr, use_container_width=True)

    # ---- PHASE DE PRÉDICTION ----
    st.write("## 🔍 Phase de Prédiction")

    # Sélectionner les caractéristiques d'entrée pour la prédiction
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data = {}
    for feature in features:
        if feature in data.columns:
            input_data[feature] = st.sidebar.slider(f"{feature}", min_value=int(data[feature].min()), max_value=int(data[feature].max()), value=int(data[feature].mean()))

    # Prédiction avec le modèle
    if st.sidebar.button("Prédire si vous êtes affecté"):
        input_df = pd.DataFrame([input_data])
        if hasattr(model, 'predict'):
            prediction = model.predict(input_df)
            result = "🛑 Affecté" if prediction[0] == 1 else "✅ Non affecté"
            st.subheader(f"### Résultat de la prédiction : {result}")
        else:
            st.error("Le modèle n'a pas été chargé correctement. Vérifiez le fichier du modèle.")

else:
    st.write("Veuillez charger un fichier CSV pour commencer.")
