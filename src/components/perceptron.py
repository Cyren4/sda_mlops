import streamlit as st
import pandas as pd
import numpy as np
import os
##
from arize.utils.types import ModelTypes, Environments
##
from dotenv import load_dotenv
load_dotenv()



def perceptron(run_ID, p, arize_client, schema):
    """Displays the main page of the app."""
    st.header("Perceptron de Rosenblatt - Analyse des performances")

    rf_page = st.sidebar.radio("Subsection", ["Performance de Perceptron de Rosenblatt", "Prédiction de Perceptron de Rosenblatt"])

    if rf_page == "Performance de Perceptron de Rosenblatt":
        # Charger les métriques sauvegardées
        st.subheader("Performance de Perceptron de Rosenblatt")
        artifacts_path = f"{os.getcwd()}/src/models/mlruns_perceptron/986589959954045561/{run_ID}/artifacts" 

        # Affichage des images enregistrées dans MLflow
        st.subheader("Matrice de Confusion")
        cm_path = f"{artifacts_path}/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path)
        else:
            st.warning("Matrice de confusion introuvable : " + cm_path)

        st.subheader("Courbe ROC")
        roc_path = f"{artifacts_path}/roc_curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path)
        else:
            st.warning("Courbe ROC introuvable : " + roc_path)

        st.subheader("Courbe Précision-Rappel")
        pr_path = f"{artifacts_path}/precision_recall_curve.png"
        if os.path.exists(pr_path):
            st.image(pr_path)
        else:
            st.warning("Courbe Précision-Rappel introuvable : " + pr_path)

        st.subheader("Validation croisée sur perceptron")
        cross_val_path = f"{artifacts_path}/cross_val_perceptron.png"
        if os.path.exists(cross_val_path):
            st.image(cross_val_path)
        else:
            st.warning("Validation croisée sur perceptron introuvable : " + cross_val_path)





    elif rf_page == "Prédiction de Perceptron de Rosenblatt":
        st.title("Prédiction du défaut de paiement - Perceptron de Rosenblatt")
        st.markdown("Remplissez les informations du client pour obtenir une prédiction.")

        credit_lines_outstanding = st.number_input("Nombre de lignes de crédit en cours", min_value=0, max_value=50, value=5)
        loan_amt_outstanding = st.number_input("Montant du prêt en cours ($)", min_value=0, max_value=1000000, value=20000)
        total_debt_outstanding = st.number_input("Dette totale en cours ($)", min_value=0, max_value=5000000, value=50000)
        income = st.number_input("Revenu annuel ($)", min_value=1000, max_value=1000000, value=50000)
        years_employed = st.number_input("Années d'emploi", min_value=0, max_value=50, value=5)
        fico_score = st.slider("Score FICO", min_value=300, max_value=850, value=600)

        # Assume you have actual labels available for evaluation
        actual_label = st.number_input("Defaut de paiement (1: Yes, 0: No)", min_value=0, max_value=1, value=1)

        input_data = pd.DataFrame({
            "credit_lines_outstanding": [credit_lines_outstanding],
            "loan_amt_outstanding": [loan_amt_outstanding],
            "total_debt_outstanding": [total_debt_outstanding],
            "income": [income],
            "years_employed": [years_employed],
            "fico_score": [fico_score],
        })
        if st.button("Prédire le défaut de paiement"):
            prediction = [p.predict(np.array(input_data)[i]) for i in range(len(input_data))]
            resultat = "Risque de défaut de paiement !" if prediction[0] == 1 else "Aucun risque détecté."
            st.subheader("Résultat de la prédiction")
            st.write(resultat)

               # Log the prediction to Arize
            timestamp = pd.Timestamp.now()
            
            # Log the prediction to Arize
            data = {
                "customer_id": [str(timestamp.timestamp())],  # Unique ID for each prediction
                "timestamp": [timestamp],
                "credit_lines_outstanding": [credit_lines_outstanding],
                "loan_amt_outstanding": [loan_amt_outstanding],
                "total_debt_outstanding": [total_debt_outstanding],
                "income": [income],
                "years_employed": [years_employed],
                "fico_score": [fico_score],
                "prediction_label": [prediction[0]],
                "actual_label": [actual_label]             
            }
            dataframe = pd.DataFrame(data)
            
            try: 
                response = arize_client.log(
                    dataframe = dataframe,
                    model_id="perceptron_de_rosenblatt",
                    model_version="v1",
                    model_type=ModelTypes.SCORE_CATEGORICAL,
                    environment=Environments.PRODUCTION,
                    #features=features,
                    #prediction_label = [int(prediction[0])],
                    schema=schema
                )

                if response.status_code != 200:
                    print(f"Failed to log data to Arize: {response.text}")
                else:
                    print("Successfully logged data to Arize")
            except Exception as e: # pylint: disable=broad-except
                print(f"An error occured: {e}")

