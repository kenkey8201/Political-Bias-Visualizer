import umap
import pandas as pd
from vectorize import vectorize_text
import matplotlib.pyplot as plt
import joblib

def reduce_dimensions(vectors, n_neighbors=15, min_dist=0.1):
    """Reduce dimensions using UMAP"""
    reducer = umap.UMAP(
        n_components=2, 
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    embedding = reducer.fit_transform(vectors)
    joblib.dump(reducer, 'umap_reducer.pkl')
    return embedding

def create_visualization():
    # Load and process data
    df = pd.read_csv('dataset/dataset.csv')
    vectors, _ = vectorize_text(df['text'])
    
    # Reduce dimensions
    embedding = reduce_dimensions(vectors)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    colors = {'Left': 'blue', 'Center': 'green', 'Right': 'red'}
    for label in df['label'].unique():
        idx = df['label'] == label
        plt.scatter(
            embedding[idx, 0], 
            embedding[idx, 1],
            c=colors[label],
            label=label,
            alpha=0.6
        )
    
    plt.title('Political Bias Visualization')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.savefig('bias_visualization.png')
    plt.close()

if __name__ == "__main__":
    create_visualization()