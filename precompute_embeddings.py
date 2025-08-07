import numpy as np
import pandas as pd
import joblib
from vectorize import vectorize_text

# Load dataset
df = pd.read_csv('dataset/dataset.csv')

# Vectorize text
vectors, _ = vectorize_text(df['text'])

# Load UMAP reducer
reducer = joblib.load('umap_reducer.pkl')

# Transform vectors to UMAP space
dataset_embedding = reducer.transform(vectors)

# Save embeddings
np.save('dataset_embedding.npy', dataset_embedding)

print(f"Saved embeddings for {len(df)} samples to dataset_embedding.npy")