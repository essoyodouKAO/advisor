import streamlit as st
import pandas as pd 

st.set_page_config(
    page_title="My Streamlit App",
    page_icon="🚀" ,
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Welcome to My Streamlit App 🚀")
st.write("This is a sample Streamlit application with customized page settings.")



st.title("🤖 DbAdvisor")
st.caption("Je t'aide à choisir la meilleure base de données pour ton projet.") 
st.header("Database Recommendation Chatbot")

df = pd.read_csv(r"C:\Users\emile\Downloads\loan_data(in).csv")
st.dataframe(df.head())