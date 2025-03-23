import streamlit as st
import pandas as pd
import os

def arbre(run_ID, tree):
    """Displays the main page of the app."""
    st.header("Arbre de décision - Analyse des performances")

    rf_page = st.sidebar.radio("Subsection", ["Performance de Arbre de décision", "Prédiction de Arbre de décision"])

    if rf_page == "Performance de Arbre de décision":
        # Charger les métriques sauvegardées
        artifacts_path = f"{os.getcwd()}/src/models/mlruns_tree/961169546191155350/{run_ID}/artifacts" 
        metrics_path = f"{os.getcwd()}/src/models/mlruns_tree/961169546191155350/{run_ID}/metrics"


        st.subheader("Tree structure")
        roc_path = f"{artifacts_path}/arbre.png"
        if os.path.exists(roc_path):
            st.image(roc_path)
        else:
            st.warning("Structure de l'arbre introuvable : " + roc_path)

        # Affichage des images enregistrées dans MLflow
        st.subheader("Matrice de Confusion")
        cm_path = f"{artifacts_path}/confusion_matrix_tree.png"
        if os.path.exists(cm_path):
            st.image(cm_path)
        else:
            st.warning("Matrice de confusion introuvable : " + cm_path)

        st.subheader("Courbe ROC")
        roc_path = f"{artifacts_path}/roc_curve_tree.png"
        if os.path.exists(roc_path):
            st.image(roc_path)
        else:
            st.warning("Courbe ROC introuvable : " + roc_path)

        st.subheader("Courbe Précision-Rappel")
        pr_path = f"{artifacts_path}/precision_recall_curve_tree.png"
        if os.path.exists(pr_path):
            st.image(pr_path)
        else:
            st.warning("Courbe Précision-Rappel introuvable : " + pr_path)

        st.subheader("Validation croisée sur Arbre de décision")
        cross_val_path = f"{artifacts_path}/cross_val_perceptron_tree.png"
        if os.path.exists(cross_val_path):
            st.image(cross_val_path)
        else:
            st.warning("Validation croisée sur arbre de décision introuvable : " + cross_val_path)





    elif rf_page == "Prédiction de Arbre de décision":
        st.title("Prédiction du défaut de paiement - Arbre de décision")
        st.markdown("Remplissez les informations du client pour obtenir une prédiction.")

        credit_lines_outstanding = st.number_input("Nombre de lignes de crédit en cours", min_value=0, max_value=50, value=5)
        loan_amt_outstanding = st.number_input("Montant du prêt en cours ($)", min_value=0, max_value=1000000, value=20000)
        total_debt_outstanding = st.number_input("Dette totale en cours ($)", min_value=0, max_value=5000000, value=50000)
        income = st.number_input("Revenu annuel ($)", min_value=1000, max_value=1000000, value=50000)
        years_employed = st.number_input("Années d'emploi", min_value=0, max_value=50, value=5)
        fico_score = st.slider("Score FICO", min_value=300, max_value=850, value=600)
        total_debt_outstanding_ratio = total_debt_outstanding

        input_data = pd.DataFrame({
            "credit_lines_outstanding": [credit_lines_outstanding],
            "loan_amt_outstanding": [loan_amt_outstanding],
            "total_debt_outstanding": [total_debt_outstanding],
            "income": [income],
            "years_employed": [years_employed],
            "fico_score": [fico_score],
            "total_debt_outstanding_ratio": [total_debt_outstanding_ratio],
        })
        import numpy as np

        if st.button("Prédire le défaut de paiement"):
            prediction = [tree.predict(np.array(input_data)[i]) for i in range(len(input_data))]
            resultat = "Risque de défaut de paiement !" if prediction[0] == 1 else "Aucun risque détecté."
            st.subheader("Résultat de la prédiction")
            st.write(resultat)
