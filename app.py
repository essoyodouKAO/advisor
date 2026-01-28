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

# 1. Configuration de la page
st.set_page_config(page_title="Loan Predictor Pro", page_icon="🏦", layout="wide")

# 2. Chargement et Nettoyage des données
@st.cache_data
def load_and_clean_data():
    # UTILISE LE CHEMIN COMPLET AVEC 'r' DEVANT
    path = "loan_data.csv"
    try:
        data = pd.read_csv(path, encoding='latin-1')
        
        # Nettoyage rapide pour l'entraînement
        df_clean = data.copy()
        
        # Supprimer l'ID s'il existe (inutile pour le ML)
        if 'Loan_ID' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Loan_ID'])
        
        # Remplissage des valeurs manquantes
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Encodage des variables texte en nombres
        le = LabelEncoder()
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = le.fit_transform(df_clean[col])
            
        return data, df_clean
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

df_raw, df_ml = load_and_clean_data()

# 3. Sidebar (Sélecteur de modèle)
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
    # --- SECTION MÉTRIQUES ---
    c1, c2, c3, c4 = st.columns(4)
    
    # Total Dossiers
    c1.metric("Total Dossiers", len(df_raw))
    
    # Taux d'Approbation (sécurisé)
    if 'Loan_Status' in df_raw.columns:
        app_rate = (df_raw['Loan_Status'].isin(['Y', 1])).mean() * 100
        c2.metric("Taux d'Approbation", f"{app_rate:.1f}%")
    
    # Prêt Moyen
    if 'LoanAmount' in df_raw.columns:
        avg_loan = df_raw['LoanAmount'].mean()
        c3.metric("Prêt Moyen", f"{avg_loan:.1f}k$")
    
    # Revenu Moyen
    income_col = 'ApplicantIncome' if 'ApplicantIncome' in df_raw.columns else df_raw.select_dtypes(include=[np.number]).columns[0]
    c4.metric("Revenu Moyen", f"{df_raw[income_col].mean():.0f}$")

    # --- ONGLETS ---
    tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🔮 Entraînement", "📈 Visualisation"])

    with tab1:
        st.subheader("Aperçu des données brutes")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.write("Statistiques globales :")
        st.write(df_raw.describe())

    with tab2:
        st.subheader("Entraînement en temps réel")
        
        if 'Loan_Status' in df_ml.columns:
            X = df_ml.drop('Loan_Status', axis=1)
            y = df_ml['Loan_Status']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if st.button("🚀 Lancer l'apprentissage"):
                with st.spinner('Le modèle analyse les données...'):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    acc = accuracy_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)

                    st.success(f"Modèle {choix_modele} entraîné !")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Précision (Accuracy)", f"{acc:.2%}")
                    m2.metric("Rappel (Recall)", f"{rec:.2%}")
                    m3.metric("F1-Score", f"{f1:.2%}")
        else:
            st.error("La colonne cible 'Loan_Status' est absente.")

    with tab3:
        st.subheader("Analyses Graphiques")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            if 'LoanAmount' in df_raw.columns:
                fig1, ax1 = plt.subplots()
                sns.histplot(df_raw['LoanAmount'].dropna(), kde=True, color="skyblue", ax=ax1)
                ax1.set_title("Distribution des montants de prêts")
                st.pyplot(fig1)

        with col_g2:
            if 'LoanAmount' in df_raw.columns and income_col in df_raw.columns:
                fig2, ax2 = plt.subplots()
                sns.scatterplot(data=df_raw, x=income_col, y='LoanAmount', hue='Loan_Status' if 'Loan_Status' in df_raw.columns else None, ax=ax2)
                ax2.set_title("Revenu vs Montant du Prêt")
                st.pyplot(fig2)
else:
    st.error("Impossible de trouver le fichier CSV. Vérifiez le chemin : C:\\Users\\emile\\Downloads\\loan_data(in).csv")