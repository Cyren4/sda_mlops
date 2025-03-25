import streamlit as st

# === PAGE 1 : INTRODUCTION ===
def introduction():
    """Displays the INTRODUCTION page of the app."""

    st.header("🔎 Predicting Loan Defaults in Retail Banking")
    
    st.markdown("""
    ### **📌 Contexte**
    Le secteur bancaire de détail connaît une augmentation des taux de défaut sur les prêts personnels. Étant donné que ces prêts représentent une source de revenus importante, il est crucial de pouvoir **prédire ces défauts**.  
    Un **défaut** survient lorsqu'un emprunteur cesse d'effectuer les paiements requis sur sa dette.

    ### **🎯 Objectif**
    L'équipe de **gestion des risques** cherche à anticiper ces défauts pour minimiser les pertes financières.  
    L'objectif principal est de construire un **modèle prédictif** qui estime la **probabilité de défaut** pour chaque client.  
    Des prédictions précises permettront à la banque d’**optimiser ses décisions** et d’**allouer efficacement son capital**.

    ---

    ### **⚙️ Prétraitement des données**
    Avant d'entraîner nos modèles, nous avons appliqué un **prétraitement rigoureux** aux données :
    
    - 📊 **Normalisation des variables numériques** :  
        Les variables sont mises à l'échelle avec la **Standardisation (StandardScaler)** pour éviter les biais liés aux différences d'unités.
    
    - ⚖️ **Rééquilibrage des classes (Oversampling)** :  
        Le dataset est déséquilibré (peu de défauts). Pour corriger cela, nous utilisons **SMOTE (Synthetic Minority Over-sampling Technique)** afin de générer des données synthétiques et équilibrer la distribution des classes.
    
    ```python
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    # Normalisation des variables
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Rééquilibrage avec SMOTE
    smote = SMOTE(sampling_strategy=1, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    ```
    
    ---

    ### **🧠 Approche Machine Learning**
    Nous allons entraîner et comparer **trois modèles** de classification pour prédire le risque de défaut :

    - 🌲 **Random Forest** : Algorithme d'ensemble robuste et performant pour les données tabulaires.
    - 📏 **Decision Tree** : Arbre de décision simple, facile à interpréter.
    - 🔴 **Perceptron** : Algorithme inspiré des réseaux de neurones, utile pour les problèmes linéairement séparables.

    Sélectionnez un modèle dans la **barre latérale** pour explorer ses performances et faire des prédictions. 📊🔍
    """)