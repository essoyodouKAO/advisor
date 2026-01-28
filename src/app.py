import streamlit as st
import pandas as pd
from engine import AnalyticsEngine # On importe notre moteur
import os

st.set_page_config(page_title="Opensee Analytics Layer", layout="wide")

# Simulation de la "Couche Produit"
def main():
    st.title("📊 Financial Analytics Dashboard")
    
    path = "data/loan_data.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # Appel du moteur de calcul (Couche Engine)
        metrics = AnalyticsEngine.calculate_portfolio_metrics(df)
        
        # Affichage des KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Exposition Totale", f"{metrics['total_exposure']:,} $")
        c2.metric("Ratio Endettement Moyen", f"{metrics['mean_debt_ratio']:.2f}")
        c3.metric("Volatilité du Risque", f"{metrics['risk_volatility']:.2f}")
        
        st.subheader("Visualisation du Stress Test (Scénario -20% revenus)")
        stressed_data = AnalyticsEngine.stress_test(df)
        st.line_chart(stressed_data[:50]) # Affiche les 50 premiers
    else:
        st.error("CSV introuvable dans /data")

if __name__ == "__main__":
    main()