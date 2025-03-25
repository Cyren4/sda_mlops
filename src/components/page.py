import iads.Classifiers
import streamlit as st
import os
import mlflow
from components.introduction import introduction
from components.random_forest import  random_forest
from components.lstm import lstm
from components.perceptron import perceptron
from components.tree import arbre
##
from arize.pandas.logger import Client, Schema


ARIZE_SPACE_ID=os.getenv("SPACE_ID")
ARIZE_API_KEY = os.getenv("API_KEY") 

# Define the schema for your data
schema = Schema(
    prediction_id_column_name="customer_id",
    timestamp_column_name="timestamp",
    feature_column_names=["credit_lines_outstanding", "years_employed", "fico_score", "total_debt_outstanding", "income", "loan_amt_outstanding"],
    prediction_label_column_name="prediction_label",
    actual_label_column_name="actual_label"
)

def get_runID(page):
    if page == "🌲 Modèle Random Forest":
        return "e4631371ac2544b587164e4f9074f25a" , "/src/mlruns/0/e4631371ac2544b587164e4f9074f25a/artifacts/random_forest_model"
    elif page == "🌹 Perceptron de Rosenblatt":
        return "c4652bfb24de4838b80406966bb74891" , "/src/models/mlruns_perceptron/986589959954045561/c4652bfb24de4838b80406966bb74891/artifacts/perceptron" 
    elif page == "🌲 Arbre de Décision":
        return "d53bb01c8a9f4c648839817a5a3837ea" , "/src/models/mlruns_tree/961169546191155350/d53bb01c8a9f4c648839817a5a3837ea/artifacts/tree"
    # LSTM
    return "e4631371ac2544b587164e4f9074f25a" , "/src/mlruns/0/e4631371ac2544b587164e4f9074f25a/artifacts/random_forest_model" 


def select_page(page):
    """Select the page to display based on the user's selection."""
    # Get dynamic path
    pwd = os.getcwd() 

    # Initialize Arize client with your space key and api key
    arize_client = Client(space_id=ARIZE_SPACE_ID, api_key=ARIZE_API_KEY)

    if page == "🏦 Introduction":
        introduction()
    elif page == "📈 Modèle LSTM":
        lstm()
    elif page == "🏕️ Modèle Random Forest":
        run_ID, model_URI = get_runID(page)
        random_forest(run_ID, load_model(pwd + model_URI), arize_client, schema)
    elif page == "🌹 Perceptron de Rosenblatt":
        import iads as iads
        p = iads.Classifiers.ClassifierPerceptron(6, learning_rate=0.01, init=True)
        run_ID, model_URI = get_runID(page)
        perceptron(run_ID, p, arize_client, schema)
    elif page == "🌲 Arbre de Décision":
        import iads as iads
        a = iads.Classifiers.ClassifierArbreNumerique(epsilon=0.0, input_dimension=7)

        run_ID, model_URI = get_runID(page)
        # arbre(run_ID, a)
        arbre(run_ID, load_model(pwd + model_URI), arize_client, schema)

# === CHARGER LE MODÈLE RANDOM FOREST MLflow ===
@st.cache_resource
def load_model(model_URI):
    return mlflow.pyfunc.load_model(model_URI)