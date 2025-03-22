import streamlit as st
import pandas as pd
import os

# === PAGE 3 : MODÈLE RANDOM FOREST ===
def random_forest(run_ID, rf_model):
    """Displays the main page of the app."""
    st.header("🌲 Modèle Random Forest - Analyse des performances")

    rf_page = st.sidebar.radio("Sous-section", ["📊 Performance Random Forest", "🤖 Prédiction Random Forest"])

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

