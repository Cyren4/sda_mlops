import streamlit as st

# === PAGE 1 : INTRODUCTION ===
def introduction():
    """Displays the INTRODUCTION page of the app."""

    st.header("🔎 Predicting Loan Defaults in Retail Banking")
    
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