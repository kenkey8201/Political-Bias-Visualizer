import streamlit as st
import pandas as pd
import numpy as np
import joblib
from vectorize import vectorize_text
import matplotlib.pyplot as plt

# Load models
@st.cache_resource
def load_models():
    vectorizer = joblib.load('vectorizer.pkl')
    classifier = joblib.load('classifier.pkl')
    reducer = joblib.load('umap_reducer.pkl')
    return vectorizer, classifier, reducer

# Load dataset and precomputed embeddings
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/dataset.csv')
    # Load precomputed embeddings if they exist, otherwise compute them
    try:
        dataset_embedding = np.load('dataset_embedding.npy')
    except FileNotFoundError:
        # Compute embeddings for the entire dataset
        vectors, vectorizer = vectorize_text(df['text'])
        reducer = joblib.load('umap_reducer.pkl')
        dataset_embedding = reducer.transform(vectors)
        np.save('dataset_embedding.npy', dataset_embedding)
    return df, dataset_embedding

# Main app
def main():
    st.title("Political Bias Visualizer")
    st.write("Classify and visualize political leaning of text content")
    
    # Load models and data
    vectorizer, classifier, reducer = load_models()
    df, dataset_embedding = load_data()
    
    # Text input
    user_input = st.text_area("Enter text to analyze:", height=150)
    
    if st.button("Analyze"):
        if user_input:
            # Vectorize input
            vectors, _ = vectorize_text([user_input], vectorizer)
            
            # Predict
            prediction = classifier.predict(vectors)[0]
            proba = classifier.predict_proba(vectors)[0]
            confidence = np.max(proba)
            
            # Display results
            st.write(f"**Predicted Bias:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            # Transform user input to UMAP space
            user_embedding = reducer.transform(vectors)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot existing data
            colors = {'Left': 'blue', 'Center': 'green', 'Right': 'red'}
            for label in df['label'].unique():
                idx = df['label'] == label
                ax.scatter(
                    dataset_embedding[idx, 0],  # Use precomputed embeddings
                    dataset_embedding[idx, 1],
                    c=colors[label],
                    label=label,
                    alpha=0.3
                )
            
            # Plot user input
            ax.scatter(
                user_embedding[0, 0],
                user_embedding[0, 1],
                c='black',
                s=200,
                marker='*',
                edgecolors='white',
                label='Your Input'
            )
            
            ax.set_title('Political Bias Visualization')
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Please enter text to analyze")

if __name__ == "__main__":
    main()