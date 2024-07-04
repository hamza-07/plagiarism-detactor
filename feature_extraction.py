# feature_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(texts):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    return features, vectorizer
