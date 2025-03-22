import streamlit as st
import os
import mlflow
from components.introduction import introduction
from components.random_forest import  random_forest
from components.lstm import lstm


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