import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')  # <-- AJOUTE CETTE LIGNE
nltk.download('stopwords')


# Téléchargement des composants NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')

# Notre dictionnaire de connaissances (Expertise)
KNOWLEDGE = {
    "sql": "PostgreSQL est le meilleur choix pour les données structurées et les relations complexes.",
    "nosql": "MongoDB est idéal pour la flexibilité (documents JSON) et la scalabilité horizontale.",
    "vitesse": "Redis est une base de données en mémoire parfaite pour le cache et la rapidité extrême.",
    "texte": "Elasticsearch est la référence pour la recherche textuelle avancée.",
    "graphe": "Neo4j est parfait si tes données sont connectées comme un réseau social.",
}




def get_recommendation(user_text):
    # 1. Nettoyage
    tokens = word_tokenize(user_text.lower())
    stop_words = set(stopwords.words('french'))
    words = [w for w in tokens if w.isalnum() and w not in stop_words]
    
    # 2. Analyse des mots-clés
    suggestions = []
    for word in words:
        if word in KNOWLEDGE:
            suggestions.append(KNOWLEDGE[word])
    
    # 3. Réponse intelligente
    if suggestions:
        # On enlève les doublons et on joint les réponses
        return " ".join(list(set(suggestions)))
    else:
        return "Je n'ai pas trouvé de mot-clé technique (vitesse, sql, texte...). Peux-tu me donner plus de détails sur tes données ?"