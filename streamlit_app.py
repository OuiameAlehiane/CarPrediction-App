import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Retapez le chemin du fichier manuellement sans copier-coller
df = pd.read_csv("CAR_CLEAN.csv")

# Loading ML model
model = load("xgboost_pipeline.joblib")

# Setting page configuration
st.set_page_config(page_title="Predictions des Prix de Voitures d'Occasions", page_icon="🚗", layout="centered")
st.markdown("<div style='background-color:#219C90; border-radius:50px; align-items:center; justify-content: center;'>\
            <h1 style='text-align:center; color:white;'>Predictions des Prix de Voitures d'Occasions</h1></div>", unsafe_allow_html=True)
            


col1 = st.container()
with col1:
    st.image("VehiculeSale.gif", use_column_width=True)
    selected_marque = st.selectbox("Choisissez la Marque de Votre Voiture", options=df['Marque'].unique())
    filtered_models = df[df['Marque'] == selected_marque]['Modèle'].unique()
    selected_model = st.selectbox("Choisissez le Modèle de Votre Voiture", options=filtered_models)
    
    available_years = df[(df['Marque'] == selected_marque) & (df['Modèle'] == selected_model)]['Année-Modèle'].dropna().unique()
    available_years = sorted(available_years.astype(int))
    year = st.select_slider("Choisissez l'Année du Modèle :", options=available_years, value=available_years[0] if available_years else 2000)
    available_ages = df[(df['Marque'] == selected_marque) & (df['Modèle'] == selected_model) & (df['Année-Modèle'] == year)]['Age_de_voiture'].unique()
    selected_age = st.selectbox("Choisissez l'Âge de Votre Voiture", options=sorted(available_ages))
    fuel_type = st.radio("Choisissez Votre Type de Carburant", options=df["Type de carburant"].unique())
    suspension_type = st.radio("Choisissez Votre Boite de Vitesses", options=df["Boite de vitesses"].unique())
    kms_driven = st.number_input("Entrez le Kilométrage :", min_value=500, max_value=475000, step=1000, value=500)
    puissance_fiscale = st.number_input("Choisissez la Puissance Fiscale :", min_value=4, max_value=41, step=1, value=4)

pred = st.button("Estimation", use_container_width=True)

if pred:
    data = {
        'Marque': selected_marque,
        'Kilométrage': kms_driven,
        'Année-Modèle': year,
        'Modèle': selected_model,
        'Type de carburant': fuel_type,
        'Puissance fiscale': puissance_fiscale,
        'Boite de vitesses': suspension_type,
        'Age_de_voiture': selected_age
    }

    input_df = pd.DataFrame(data, index=[0])  # Ici, nous utilisons [0] comme index pour une seule ligne de données

    df_encoded = pd.get_dummies(input_df, columns=['Marque', 'Modèle', 'Boite de vitesses', 'Type de carburant'],
                                prefix=['Marque', 'Modèle', 'Boite de vitesses', 'Type de carburant'])
                                
    # Liste des features attendues par le modèle
    model_features =  ['Kilométrage', 'Année-Modèle', 'Puissance fiscale', 'Age_de_voiture', 'Marque_Audi', 'Marque_BMW', 'Marque_Citroen',  'Marque_Dacia', 'Marque_Fiat', 'Marque_Ford', 'Marque_Hyundai', 'Marque_Kia', 'Marque_Land Rover', 'Marque_Mercedes-Benz', 'Marque_Nissan', 'Marque_Opel', 'Marque_Peugeot', 'Marque_Renault', 'Marque_Toyota', 'Marque_Volkswagen', 'Modèle_Automatique', 'Modèle_Manuelle', 'Boite de vitesses_Diesel', 'Boite de vitesses_Electrique', 'Boite de vitesses_Essence', 'Boite de vitesses_Hybride', 'Boite de vitesses_LPG', 'Type de carburant_19', 'Type de carburant_190', 'Type de carburant_206', 'Type de carburant_206+', 'Type de carburant_207', 'Type de carburant_208', 'Type de carburant_220', 'Type de carburant_250', 'Type de carburant_3008', 'Type de carburant_301', 'Type de carburant_306', 'Type de carburant_307', 'Type de carburant_308', 'Type de carburant_407', 'Type de carburant_500', 'Type de carburant_508', 'Type de carburant_A3', 'Type de carburant_A4', 'Type de carburant_Accent', 'Type de carburant_Astra', 'Type de carburant_Berlingo', 'Type de carburant_C3', 'Type de carburant_C4', 'Type de carburant_Caddy', 'Type de carburant_Classe A', 'Type de carburant_Classe C', 'Type de carburant_Classe E', 'Type de carburant_Clio', 'Type de carburant_Corolla', 'Type de carburant_Corsa', 'Type de carburant_Doblo', 'Type de carburant_Dokker', 'Type de carburant_Duster', 'Type de carburant_EXPRESS', 'Type de carburant_FIORINO', 'Type de carburant_Fiesta', 'Type de carburant_Focus', 'Type de carburant_GOLF 4', 'Type de carburant_GOLF 5', 'Type de carburant_GOLF 6', 'Type de carburant_GOLF 7', 'Type de carburant_Jetta', 'Type de carburant_Kangoo', 'Type de carburant_Kuga', 'Type de carburant_Laguna', 'Type de carburant_Logan', 'Type de carburant_Megane', 'Type de carburant_Megane 3', 'Type de carburant_PASSAT CC', 'Type de carburant_Palio', 'Type de carburant_Partner', 'Type de carburant_Passat', 'Type de carburant_Picanto', 'Type de carburant_Polo', 'Type de carburant_Punto', 'Type de carburant_Q5', 'Type de carburant_Qashqai', 'Type de carburant_R19', 'Type de carburant_RAV 4', 'Type de carburant_Range Rover Evoque', 'Type de carburant_Range Rover Sport', 'Type de carburant_Sandero', 'Type de carburant_Santa Fe', 'Type de carburant_Scenic', 'Type de carburant_Serie 1', 'Type de carburant_Serie 3', 'Type de carburant_Serie 5', 'Type de carburant_Siena', 'Type de carburant_Sportage', 'Type de carburant_Tiguan', 'Type de carburant_Touareg', 'Type de carburant_Touran', 'Type de carburant_Transit', 'Type de carburant_Tucson', 'Type de carburant_Yaris', 'Type de carburant_i 10', 'Type de carburant_i 30', 'Type de carburant_ix 35', 'Type de carburant_megane_4', 'Type de carburant_sandero_stepway']

    df_template = pd.DataFrame(0, index=np.arange(len(df_encoded)), columns=model_features)
    for column in df_encoded.columns:
        if column in df_template.columns:
            df_template[column] = df_encoded[column]
    
    prediction = model.predict(df_template)[0]
    if prediction < 0:
        st.error("Predicted Price is Below Zero, Please select Valid Inputs.", icon="⚠️")
    else:
        st.success(f"Le Prix de Votre Voiture est : {int(np.round(prediction))} Dh", icon="✅")

