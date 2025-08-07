from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd

def vectorize_text(texts, vectorizer=None):
    """Convert text to TF-IDF vectors"""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        vectors = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, 'vectorizer.pkl')
    else:
        vectors = vectorizer.transform(texts)
    return vectors, vectorizer

if __name__ == "__main__":
    # Sample usage
    df = pd.read_csv('dataset/dataset.csv')
    vectors, _ = vectorize_text(df['text'])
    print(f"Vectorized shape: {vectors.shape}")