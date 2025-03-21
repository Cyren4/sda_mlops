import streamlit as st
import pandas as pd
import mlflow.pyfunc
import os

# === PAGE 1 : INTRODUCTION ===
def introduction():
    """Displays the INTRODUCTION page of the app."""

    st.title("🏦 Banking MLOps : Predicting Loan Defaults in Retail Banking")
    
    st.markdown("""
    ## **Contexte**
    Le secteur bancaire de détail connaît une augmentation des taux de défaut sur les prêts personnels. Étant donné que ces prêts représentent une source de revenus importante, il est crucial de pouvoir prédire ces défauts. Un défaut survient lorsqu'un emprunteur cesse d'effectuer les paiements requis sur sa dette.
    
    ## **Objectif**
    Notre équipe de gestion des risques analyse le portefeuille de prêts pour prévoir les défauts potentiels et estimer les pertes attendues. L'objectif principal est de construire un modèle prédictif qui estime la **probabilité de défaut** pour chaque client. Des prédictions précises permettront à la banque d'allouer efficacement son capital et de maintenir sa stabilité financière.
    
    ## **Approche Machine Learning**
    Nous allons entraîner et comparer **deux modèles** :
    - 📈 **LSTM (Long Short-Term Memory)** : Approche Deep Learning pour les séries temporelles financières.
    - 🌲 **Random Forest** : Modèle robuste d'ensemble pour les données structurées.
    
    Sélectionnez un modèle dans la barre latérale pour explorer ses performances et faire des prédictions.
    """)

# === PAGE 2 : MODÈLE LSTM ===
def lstm():
    """Displays the main page of the app."""
   
    st.title("📈 Modèle LSTM - Analyse des performances")

    lstm_page = st.sidebar.radio("Sous-section", ["📊 Performance LSTM", "🤖 Prédiction LSTM"])

    if lstm_page == "📊 Performance LSTM":
        st.markdown("🚧 **Page en construction** : Les performances du modèle LSTM seront bientôt disponibles.")

    elif lstm_page == "🤖 Prédiction LSTM":
        st.title("🤖 Prédiction du défaut de paiement - Modèle LSTM")
        st.markdown("🚧 **Page en construction** : La prédiction avec le modèle LSTM sera bientôt disponible.")

# === PAGE 3 : MODÈLE RANDOM FOREST ===
def random_forest(run_ID, rf_model):
    """Displays the main page of the app."""
    st.title("🌲 Modèle Random Forest - Analyse des performances")

    rf_page = st.sidebar.radio("Sous-section", ["📊 Performance Random Forest", "🤖 Prédiction Random Forest"])

    if rf_page == "📊 Performance Random Forest":
        # Charger les métriques sauvegardées
        st.subheader("📊 Performance du Modèle Random Forest")
        metrics_path = f"{run_ID}/metrics"
        # artifacts_path = f"{run_ID}/artifacts"
        artifacts_path = f"/Users/cyrena/Desktop/2024_Data_course/mlops_tp/sda_mlops/images"

        if os.path.exists(f"{metrics_path}/accuracy"):
            
            # Lire et afficher l'accuracy
            try:
                with open(f"{metrics_path}/accuracy", "r") as file:
                    content = file.read().strip()
                    # Récupérer la deuxième valeur 
                    accuracy = float(content.split()[1])  # Sépare par espace et prend la deuxième valeur
                    st.metric("🎯 Accuracy", f"{accuracy:.2%}")
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du chargement de l'accuracy : {e}")

        else:
            st.write("❌ Le fichier accuracy est introuvable.")

        # Affichage des images enregistrées dans MLflow
        st.subheader("📌 Matrice de Confusion")
        cm_path = f"{artifacts_path}/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Matrice de Confusion")
        else:
            st.warning("⚠️ Matrice de confusion introuvable.")

        st.subheader("📈 Courbe ROC")
        roc_path = f"{artifacts_path}/roc_curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Courbe ROC")
        else:
            st.warning("⚠️ Courbe ROC introuvable.")

        st.subheader("📉 Courbe Précision-Rappel")
        pr_path = f"{artifacts_path}/precision_recall_curve.png"
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Courbe Précision-Rappel")
        else:
            st.warning("⚠️ Courbe Précision-Rappel introuvable.")

        st.subheader("💡 Importance des Features")
        fi_path = f"{artifacts_path}/feature_importances.png"
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Importance des Features")
        else:
            st.warning("⚠️ Importance des Features introuvable.")

    elif rf_page == "🤖 Prédiction Random Forest":
        st.title("🤖 Prédiction du défaut de paiement - Random Forest")

        st.markdown("**Remplissez les informations du client pour obtenir une prédiction.**")

        # Champs de saisie mis à jour
        credit_lines_outstanding = st.number_input("Nombre de lignes de crédit en cours", min_value=0, max_value=50, value=5)
        loan_amt_outstanding = st.number_input("Montant du prêt en cours ($)", min_value=0, max_value=1000000, value=20000)
        total_debt_outstanding = st.number_input("Dette totale en cours ($)", min_value=0, max_value=5000000, value=50000)
        income = st.number_input("Revenu annuel ($)", min_value=1000, max_value=1000000, value=50000)
        years_employed = st.number_input("Années d'emploi", min_value=0, max_value=50, value=5)
        fico_score = st.slider("Score FICO", min_value=300, max_value=850, value=600)

        # Transformer les entrées en DataFrame
        input_data = pd.DataFrame({
            "credit_lines_outstanding": [credit_lines_outstanding],
            "loan_amt_outstanding": [loan_amt_outstanding],
            "total_debt_outstanding": [total_debt_outstanding],
            "income": [income],
            "years_employed": [years_employed],
            "fico_score": [fico_score]
        })
        
        # Assurez-vous que les données ont les bons types
        input_data = input_data.astype({
            "credit_lines_outstanding": "int64",
            "loan_amt_outstanding": "float64",
            "total_debt_outstanding": "float64",
            "income": "float64",
            "years_employed": "int64",
            "fico_score": "int64"
        })

        # Bouton pour prédire
        if st.button("Prédire le défaut de paiement"):
            prediction = rf_model.predict(input_data)
            resultat = "⚠️ Risque de défaut de paiement !" if prediction[0] == 1 else "✅ Aucun risque détecté."
            st.subheader("Résultat de la prédiction")
            st.write(resultat)

