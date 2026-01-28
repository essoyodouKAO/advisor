import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os

# 1. Configuration
st.set_page_config(page_title="Loan Predictor Pro", page_icon="🏦", layout="wide")

# 2. Chargement des données
@st.cache_data
def load_and_clean_data():
    path = "loan_data.csv"
    
    # Vérification si le fichier existe sur le serveur
    if not os.path.exists(path):
        st.error(f"❌ Fichier '{path}' introuvable sur GitHub.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        data = pd.read_csv(path, encoding='latin-1')
        df_clean = data.copy()
        
        if 'Loan_ID' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Loan_ID'])
        
        # Remplissage intelligent
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Encodage
        le = LabelEncoder()
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = le.fit_transform(df_clean[col])
            
        return data, df_clean
    except Exception as e:
        st.error(f"🔥 Erreur lors de la lecture : {e}")
        return pd.DataFrame(), pd.DataFrame()

df_raw, df_ml = load_and_clean_data()

# 3. Sidebar
st.sidebar.header("⚙️ Paramètres du Modèle")
choix_modele = st.sidebar.selectbox("Algorithme :", ["Logistic Regression", "Random Forest"])

if choix_modele == "Logistic Regression":
    c_param = st.sidebar.slider("Régularisation (C)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=c_param, max_iter=1000)
else:
    n_trees = st.sidebar.slider("Nombre d'arbres", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=n_trees)

# 4. Interface Principale
st.title("🏦 Advisor : Analyse & Prédiction de Prêts")

if not df_raw.empty:
    # Métriques dynamiques
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Dossiers", len(df_raw))
    
    if 'Loan_Status' in df_raw.columns:
        app_rate = (df_raw['Loan_Status'].isin(['Y', 1])).mean() * 100
        c2.metric("Taux d'Approbation", f"{app_rate:.1f}%")
    
    if 'LoanAmount' in df_raw.columns:
        c3.metric("Prêt Moyen", f"{df_raw['LoanAmount'].mean():.1f}k$")
    
    income_col = 'ApplicantIncome' if 'ApplicantIncome' in df_raw.columns else df_raw.select_dtypes(include=[np.number]).columns[0]
    c4.metric("Revenu Moyen", f"{df_raw[income_col].mean():.0f}$")

    tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🔮 Entraînement", "📈 Visualisation"])

    with tab1:
        st.subheader("Données brutes")
        st.dataframe(df_raw.head(10), use_container_width=True)

    with tab2:
        st.subheader("Entraînement")
        if 'Loan_Status' in df_ml.columns:
            X = df_ml.drop('Loan_Status', axis=1)
            y = df_ml['Loan_Status']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if st.button("🚀 Lancer l'apprentissage"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                m2.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
                m3.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")

    with tab3:
        st.subheader("Analyses Graphiques")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig1, ax1 = plt.subplots()
            sns.histplot(df_raw['LoanAmount'].dropna(), kde=True, ax=ax1)
            st.pyplot(fig1)
        with col_g2:
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df_raw, x=income_col, y='LoanAmount', hue='Loan_Status' if 'Loan_Status' in df_raw.columns else None, ax=ax2)
            st.pyplot(fig2) 