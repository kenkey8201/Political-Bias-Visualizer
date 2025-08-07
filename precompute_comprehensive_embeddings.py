# precompute_comprehensive_embeddings.py
import numpy as np
import pandas as pd
import joblib
import re

# Custom text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load comprehensive dataset
df = pd.read_csv('dataset/comprehensive_dataset.csv')

# Preprocess text
df['processed_text'] = df['text'].apply(preprocess_text)

# Load vectorizer and UMAP reducer
vectorizer = joblib.load('vectorizer.pkl')
reducer = joblib.load('umap_reducer.pkl')

# Vectorize text
vectors = vectorizer.transform(df['processed_text'])

# Transform vectors to UMAP space
dataset_embedding = reducer.transform(vectors)

# Save embeddings
np.save('comprehensive_dataset_embedding.npy', dataset_embedding)

print(f"Saved embeddings for {len(df)} samples to comprehensive_dataset_embedding.npy")