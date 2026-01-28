import numpy as np
import pandas as pd



class AnalyticsEngine:
    """
    Couche Produit : Gère les calculs Big Data vectorisés.
    C'est ici qu'on répond aux enjeux de performance d'Opensee.
    """
    @staticmethod
    def calculate_portfolio_metrics(df):
        # Utilisation de NumPy pour la performance (Vectorisation)
        incomes = df['ApplicantIncome'].values
        loans = df['LoanAmount'].values
        
        # Calcul matriciel du ratio d'endettement
        # On évite les boucles pour traiter 1M de lignes instantanément
        debt_ratios = loans / np.where(incomes == 0, 1, incomes)
        
        metrics = {
            "mean_debt_ratio": np.mean(debt_ratios),
            "total_exposure": np.sum(loans),
            "risk_volatility": np.std(debt_ratios)
        }
        return metrics

    @staticmethod
    def stress_test(df, haircut=0.2):
        """Simule une baisse des revenus de X% (Algèbre linéaire simple)"""
        original_incomes = df['ApplicantIncome'].values
        stressed_incomes = original_incomes * (1 - haircut)
        return stressed_incomes