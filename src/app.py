# import libraries
import streamlit as st
import pandas as pd
import mlflow
import os
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier

# Import custom components
from components.header import display_header, display_context
from components.page import introduction, lstm, random_forest
# from components.context import display_context

# === CONFIGURATION DE LA PAGE ===
st.set_page_config(page_title="Banking MLOps - Prédiction du défaut de paiement", layout="wide")

def main():
    """Main function to run the Streamlit app."""
    # === CONFIGURATION MLflow === 
    # mlrun_path = "/Users/cyrena/Desktop/2024_Data_course/mlops_tp/0/"
    
    run_ID = "e4631371ac2544b587164e4f9074f25a"  # Remplace par l'ID de ton modèle correct 


    display_header()
    display_context()
    # === BARRE DE NAVIGATION ===
    st.sidebar.title("Navigation")
    main_page = st.sidebar.radio("Sélectionner une section", ["🏦 Introduction", "📈 Modèle LSTM", "🌲 Modèle Random Forest"])
    select_page(main_page, run_ID)

def select_page(page, run_ID=None):
    """Select the page to display based on the user's selection."""
    # Get dynamic path
    path_mlrun = os.getcwd() + "/src/mlruns/0/" 
    model_URI = f"{path_mlrun}{run_ID}/artifacts/random_forest_model"

    if page == "🏦 Introduction":
        introduction()
    elif page == "📈 Modèle LSTM":
        lstm()
    elif page == "🌲 Modèle Random Forest":
        random_forest(run_ID, load_model(model_URI))

# === CHARGER LE MODÈLE RANDOM FOREST MLflow ===
@st.cache_resource
def load_model(model_URI):
    return mlflow.pyfunc.load_model(model_URI)

if __name__ == "__main__":
    main()


# TODO 
# Table of content

# TODO
# Data discovery and preprocessing 

