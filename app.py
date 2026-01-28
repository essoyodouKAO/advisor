import streamlit as st
from logic import get_recommendation

st.set_page_config(page_title="DbAdvisor", page_icon="🤖")
st.title("🤖 DbAdvisor")
st.caption("Je t'aide à choisir la meilleure base de données pour ton projet.")

# Initialiser l'historique du chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée de l'utilisateur
if prompt := st.chat_input("Décris ton besoin (ex: Je veux du SQL et de la vitesse)"):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Obtenir la réponse du cerveau NLTK
    response = get_recommendation(prompt)

    # Afficher et sauvegarder la réponse du bot
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})