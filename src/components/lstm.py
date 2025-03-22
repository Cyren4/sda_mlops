import streamlit as st

# === PAGE 2 : MODÈLE LSTM ===
def lstm():
    """Displays the main page of the app."""
   
    st.header("📈 Modèle LSTM - Analyse des performances")

    lstm_page = st.sidebar.radio("Sous-section", ["📊 Performance LSTM", "🤖 Prédiction LSTM"])

    if lstm_page == "📊 Performance LSTM":
        st.markdown("🚧 **Page en construction** : Les performances du modèle LSTM seront bientôt disponibles.")

    elif lstm_page == "🤖 Prédiction LSTM":
        st.header("🤖 Prédiction du défaut de paiement - Modèle LSTM")
        st.markdown("🚧 **Page en construction** : La prédiction avec le modèle LSTM sera bientôt disponible.")
