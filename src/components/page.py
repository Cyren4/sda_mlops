import streamlit as st
import os
import mlflow
from components.introduction import introduction
from components.random_forest import  random_forest
from components.lstm import lstm
from components.perceptron import perceptron
from components.tree import arbre


def get_runID(page):
    if page == "🌲 Modèle Random Forest":
        return "e4631371ac2544b587164e4f9074f25a" , "/src/mlruns/0/e4631371ac2544b587164e4f9074f25a/artifacts/random_forest_model"
    elif page == "Perceptron de Rosenblatt":
        return "e4631371ac2544b587164e4f9074f25a" , "/src/mlruns/0/e4631371ac2544b587164e4f9074f25a/artifacts/random_forest_model" 
    elif page == "Arbre de Décision":
        return "e4631371ac2544b587164e4f9074f25a" , "/src/mlruns/0/e4631371ac2544b587164e4f9074f25a/artifacts/random_forest_model"
    return "e4631371ac2544b587164e4f9074f25a" , "/src/mlruns/0/e4631371ac2544b587164e4f9074f25a/artifacts/random_forest_model" 

def select_page(page):
    """Select the page to display based on the user's selection."""
    # Get dynamic path
    pwd = os.getcwd() 
    # default value

    if page == "🏦 Introduction":
        introduction()
    elif page == "📈 Modèle LSTM":
        lstm()
    elif page == "🏕️ Modèle Random Forest":
        run_ID, model_URI = get_runID(page)
        random_forest(run_ID, load_model(pwd + model_URI))
    elif page == "🌹 Perceptron de Rosenblatt":
        run_ID, model_URI = get_runID(page)
        perceptron(run_ID, load_model(pwd + model_URI))
    elif page == "🌲 Arbre de Décision":
        run_ID, model_URI = get_runID(page)
        arbre(run_ID, load_model(pwd + model_URI))

# === CHARGER LE MODÈLE RANDOM FOREST MLflow ===
@st.cache_resource
def load_model(model_URI):
    return mlflow.pyfunc.load_model(model_URI)