import streamlit as st
import pandas as pd
import mlflow.pyfunc
import os

# === CONFIGURATION DE LA PAGE ===
st.set_page_config(page_title="Prédiction du défaut de paiement", layout="wide")

# === CONFIGURATION MLflow ===
RUN_ID = "a1f653e1ba1645eeb33a3b62661569ef"  # Remplace par l'ID de ton modèle correct
MODEL_URI = f"/Users/yoavcohen/Desktop/sda_mlops/src/mlruns/0/{RUN_ID}/artifacts/random_forest_model"
rf_model = mlflow.pyfunc.load_model(MODEL_URI)

# === CHARGER LE MODÈLE MLflow ===
@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)

rf_model = load_model()

# === BARRE DE NAVIGATION ===
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["📊 Performance du Modèle", "🤖 Prédiction"])

# === PAGE 1 : ANALYSE DES PERFORMANCES ===
if page == "📊 Performance du Modèle":
    st.title("📊 Performance du Modèle")
    
    # Charger les métriques sauvegardées
    metrics_path = f"/Users/yoavcohen/Desktop/sda_mlops/src/mlruns/0/{RUN_ID}/metrics"
    artifacts_path = f"/Users/yoavcohen/Desktop/sda_mlops/src/mlruns/0/{RUN_ID}/artifacts"

    if os.path.exists(metrics_path):
        try:
            accuracy = float(open(f"{metrics_path}/accuracy").read().strip())
            st.metric("🎯 Accuracy", f"{accuracy:.2%}")
        except:
            st.warning("⚠️ Impossible de charger l'accuracy.")

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

# === PAGE 2 : PRÉDICTION ===
if page == "🤖 Prédiction":
    st.title("🤖 Prédiction du défaut de paiement")

    st.markdown("**Remplissez les informations du client pour obtenir une prédiction.**")

    # Champs de saisie
    age = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
    revenu_annuel = st.number_input("Revenu annuel ($)", min_value=1000, max_value=1000000, value=50000)
    score_credit = st.slider("Score de crédit", min_value=300, max_value=850, value=600)
    duree_emploi = st.number_input("Années d'emploi", min_value=0, max_value=50, value=5)
    nombre_cartes_credit = st.number_input("Nombre de cartes de crédit", min_value=0, max_value=10, value=2)
    taux_utilisation_credit = st.slider("Taux d'utilisation du crédit (%)", min_value=0, max_value=100, value=30)
    historique_paiement = st.slider("Historique de paiement (1: Très mauvais, 5: Excellent)", min_value=1, max_value=5, value=3)

    # Transformer les entrées en DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "revenu_annuel": [revenu_annuel],
        "score_credit": [score_credit],
        "duree_emploi": [duree_emploi],
        "nombre_cartes_credit": [nombre_cartes_credit],
        "taux_utilisation_credit": [taux_utilisation_credit],
        "historique_paiement": [historique_paiement]
    })

    # Bouton pour prédire
    if st.button("Prédire le défaut de paiement"):
        prediction = rf_model.predict(input_data)
        resultat = "⚠️ Risque de défaut de paiement !" if prediction[0] == 1 else "✅ Aucun risque détecté."
        st.subheader("Résultat de la prédiction")
        st.write(resultat)
