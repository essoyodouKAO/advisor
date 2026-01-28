import streamlit as st
import pandas as pd 

# 1. Configuration de la page (TOUJOURS en premier)
st.set_page_config(
    page_title="DbAdvisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Barre latérale (Sidebar) - Sortie de la colonne pour plus de clarté
st.sidebar.title("Options")
st.sidebar.write("Personnalisez votre expérience ici...")

st.sidebar.title("À propos")
version = st.sidebar.selectbox("Version de l'application", ["1.0", "1.1", "2.0"])
seuil = st.sidebar.slider("Seuil de prédiction", 0, 100, 50)

with st.sidebar.expander("Détails de l'application"):
    st.write(f"""
    **DbAdvisor** aide à choisir la meilleure base de données.
    
    - **Version**: {version}
    - **Seuil choisi**: {seuil}%
    - **Auteur**: Émile
    """)

# 3. En-tête principal
st.title("🤖 DbAdvisor")
st.caption("Je t'aide à choisir la meilleure base de données pour ton projet.") 

# 4. Chargement des données
try:
    # Ajout de encoding='latin-1' au cas où le fichier vient de Windows/Excel
    df = pd.read_csv(r"C:\Users\emile\Downloads\loan_data(in).csv", encoding='latin-1')
    st.success("Données chargées avec succès !")
    
    with st.expander("Voir les données brutes"):
        st.dataframe(df)
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier : {e}")

st.divider() # Petite ligne de séparation visuelle

# 5. Mise en page par colonnes
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Help")
    st.info("Utilisez le chatbot au centre pour obtenir des conseils personnalisés.")

with col2:
    st.header("Chatbot")
    st.write("Interface du chatbot ici...")
    user_input = st.text_input("Posez votre question à l'IA :")
    if user_input:
        st.write(f"Analyse en cours pour : {user_input}")

with col3:
    st.header("Statistiques")
    if 'df' in locals():
        st.bar_chart(df.iloc[:, 1].value_counts()) # Exemple de graphique auto
    else:
        st.write("Chargez des données pour voir les stats.")