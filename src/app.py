# import libraries
import streamlit as st

# Import custom components
from components.header import display_header, display_contributor 
from components.page import select_page

# === CONFIGURATION DE LA PAGE ===
def main():
    """Main function to run the Streamlit app."""
    # === CONFIGURATION MLflow === 
    run_ID = "e4631371ac2544b587164e4f9074f25a"  # Remplace par l'ID de ton modèle correct 
    display_header()
    # === BARRE DE NAVIGATION ===
    st.sidebar.title("Navigation")
    main_page = st.sidebar.radio("Sélectionner une section", ["🏦 Introduction", "📈 Modèle LSTM", "🌲 Modèle Random Forest"])
    select_page(main_page, run_ID)
    display_contributor()
    

if __name__ == "__main__":
    main()