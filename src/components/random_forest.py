import streamlit as st
import pandas as pd
import os
##
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
##
from dotenv import load_dotenv
load_dotenv()
import datetime

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

# === PAGE 3 : MODÈLE RANDOM FOREST ===
def random_forest(run_ID, rf_model):
    """Displays the main page of the app."""
    st.header("🌲 Modèle Random Forest - Analyse des performances")

    rf_page = st.sidebar.radio("Sous-section", ["📊 Performance Random Forest", "🤖 Prédiction Random Forest"])

    # Initialize Arize client with your space key and api key
    arize_client = Client(space_id=ARIZE_SPACE_ID, api_key=ARIZE_API_KEY)

    if rf_page == "📊 Performance Random Forest":
        # Charger les métriques sauvegardées
        st.subheader("📊 Performance du Modèle Random Forest")
        artifacts_path = f"{os.getcwd()}/src/mlruns/0/{run_ID}/artifacts" 
        metrics_path = f"{os.getcwd()}/src/mlruns/0/{run_ID}/metrics"

        if os.path.exists(f"{metrics_path}/accuracy"):
            
            # Lire et afficher l'accuracy
            try:
                with open(f"{metrics_path}/accuracy", "r", encoding="utf-8") as file:
                    content = file.read().strip()
                    # Récupérer la deuxième valeur 
                    accuracy = float(content.split()[1])  # Sépare par espace et prend la deuxième valeur
                    st.metric("🎯 Accuracy", f"{accuracy:.2%}")
            except FileNotFoundError:
                st.warning(f"⚠️ Erreur : Le fichier accuracy est introuvable à l'emplacement : {metrics_path}/accuracy")
            except IndexError:
                st.warning(f"⚠️ Erreur : Le fichier accuracy est vide ou mal formaté. Contenu : {content}")
            except ValueError:
                st.warning(f"⚠️ Erreur : La valeur d'accuracy '{accuracy}' dans le fichier n'est pas un nombre valide.")
            except PermissionError:
                st.warning(f"⚠️ Erreur : Vous n'avez pas la permission de lire le fichier : {metrics_path}/accuracy")
            except Exception as e : # pylint: disable=broad-except
                st.warning(f"⚠️ Erreur inattendue : {e}")

        else:
            st.write("❌ Le fichier accuracy est introuvable.")

        # Affichage des images enregistrées dans MLflow
        st.subheader("📌 Matrice de Confusion")
        cm_path = f"{artifacts_path}/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Matrice de Confusion")
        else:
            st.warning("⚠️ Matrice de confusion introuvable : " + cm_path)

        st.subheader("📈 Courbe ROC")
        roc_path = f"{artifacts_path}/roc_curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Courbe ROC")
        else:
            st.warning("⚠️ Courbe ROC introuvable : " + cm_path)

        st.subheader("📉 Courbe Précision-Rappel")
        pr_path = f"{artifacts_path}/precision_recall_curve.png"
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Courbe Précision-Rappel")
        else:
            st.warning("⚠️ Courbe Précision-Rappel introuvable : " + cm_path)

        st.subheader("💡 Importance des Features")
        fi_path = f"{artifacts_path}/feature_importances.png"
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Importance des Features")
        else:
            st.warning("⚠️ Importance des Features introuvable : " + cm_path)



    elif rf_page == "🤖 Prédiction Random Forest":
        st.header("🤖 Prédiction du défaut de paiement - Random Forest")

        st.markdown("**Remplissez les informations du client pour obtenir une prédiction.**")

        # Champs de saisie mis à jour
        credit_lines_outstanding = st.number_input("Nombre de lignes de crédit en cours", min_value=0, max_value=50, value=5)
        loan_amt_outstanding = st.number_input("Montant du prêt en cours ($)", min_value=0, max_value=1000000, value=20000)
        total_debt_outstanding = st.number_input("Dette totale en cours ($)", min_value=0, max_value=5000000, value=50000)
        income = st.number_input("Revenu annuel ($)", min_value=1000, max_value=1000000, value=50000)
        years_employed = st.number_input("Années d'emploi", min_value=0, max_value=50, value=5)
        fico_score = st.slider("Score FICO", min_value=300, max_value=850, value=600)

        # Assume you have actual labels available for evaluation
        actual_label = st.number_input("Defaut de paiement (1: Yes, 0: No)", min_value=0, max_value=1, value=1)

        # Transformer les entrées en DataFrame
        input_data = pd.DataFrame({
            "credit_lines_outstanding": [credit_lines_outstanding],
            "loan_amt_outstanding": [loan_amt_outstanding],
            "total_debt_outstanding": [total_debt_outstanding],
            "income": [income],
            "years_employed": [years_employed],
            "fico_score": [fico_score],
            "actual_label": [actual_label]
        })
        
        # Assurez-vous que les données ont les bons types
        input_data = input_data.astype({
            "credit_lines_outstanding": "int64",
            "loan_amt_outstanding": "float64",
            "total_debt_outstanding": "float64",
            "income": "float64",
            "years_employed": "int64",
            "fico_score": "int64",
            "actual_label": "int64"
        })

        # Bouton pour prédire
        if st.button("Prédire le défaut de paiement"):
            prediction = rf_model.predict(input_data)
            resultat = "⚠️ Risque de défaut de paiement !" if prediction[0] == 1 else "✅ Aucun risque détecté."
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
                    model_id="random_forest",
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
            except Exception as e:
                print(f"An error occured: {e}")

