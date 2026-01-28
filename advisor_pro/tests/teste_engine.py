import numpy as np
import pandas as pd
from src.engine import AnalyticsEngine

def test_stress_test():
    # On crée une donnée de test
    df = pd.DataFrame({'ApplicantIncome': [1000, 2000]})
    stressed = AnalyticsEngine.stress_test(df, haircut=0.1)
    
    # On vérifie que 1000 est devenu 900
    assert stressed[0] == 900
    print("Test Stress Test : Réussi ✅")