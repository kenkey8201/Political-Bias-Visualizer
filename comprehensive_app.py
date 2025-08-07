# comprehensive_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
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

# Load models
@st.cache_resource
def load_models():
    try:
        # Try to load the comprehensive model first
        model = joblib.load('comprehensive_classifier.pkl')
        use_comprehensive = True
    except FileNotFoundError:
        try:
            # Fall back to the improved model
            model = joblib.load('improved_classifier.pkl')
            use_comprehensive = False
        except FileNotFoundError:
            # Fall back to the original model
            vectorizer = joblib.load('vectorizer.pkl')
            classifier = joblib.load('classifier.pkl')
            use_comprehensive = False
    
    try:
        reducer = joblib.load('umap_reducer.pkl')
    except FileNotFoundError:
        reducer = None
    
    return model, use_comprehensive, reducer

# Load dataset and precomputed embeddings
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset/comprehensive_dataset.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('dataset/improved_dataset.csv')
        except FileNotFoundError:
            df = pd.read_csv('dataset/dataset.csv')
    
    # Load precomputed embeddings if they exist
    try:
        dataset_embedding = np.load('comprehensive_dataset_embedding.npy')
    except FileNotFoundError:
        try:
            dataset_embedding = np.load('improved_dataset_embedding.npy')
        except FileNotFoundError:
            try:
                dataset_embedding = np.load('dataset_embedding.npy')
            except FileNotFoundError:
                dataset_embedding = None
    
    return df, dataset_embedding

# Main app
def main():
    st.title("Political Bias Visualizer")
    st.write("Classify and visualize political leaning of text content")
    st.write("""
    **Note**: This model is for educational purposes only and may not accurately reflect real-world political biases.
    The model has been trained on a comprehensive dataset covering a wide range of political ideologies.
    """)
    
    # Add information about the political spectrum
    st.sidebar.title("Political Spectrum Information")
    st.sidebar.write("""
    **Left/Leftist**: Progressive, socialist, or communist ideologies focusing on equality, social justice, and economic redistribution.
    
    **Center/Liberal**: Moderate positions supporting incremental change, market-based solutions with some regulation, and social progress.
    
    **Right/Conservative**: Traditional values, free markets, limited government, strong national defense, and individual responsibility.
    
    **Far-Right/Fascist**: Extreme nationalism, authoritarianism, racial supremacy, and opposition to liberal democracy.
    """)
    
    # Load models and data
    model, use_comprehensive, reducer = load_models()
    df, dataset_embedding = load_data()
    
    # Text input
    user_input = st.text_area("Enter text to analyze:", height=150)
    
    if st.button("Analyze"):
        if user_input:
            # Preprocess input
            processed_input = preprocess_text(user_input)
            
            # Predict
            if use_comprehensive:
                # Using the comprehensive pipeline model
                prediction = model.predict([processed_input])[0]
                proba = model.predict_proba([processed_input])[0]
                confidence = np.max(proba)
            else:
                # Using the original or improved model
                if hasattr(model, 'named_steps'):
                    vectorizer = model.named_steps['tfidf']
                    classifier = model.named_steps['svm']
                else:
                    vectorizer = model[0]
                    classifier = model[1]
                
                vectors = vectorizer.transform([processed_input])
                prediction = classifier.predict(vectors)[0]
                proba = classifier.predict_proba(vectors)[0]
                confidence = np.max(proba)
            
            # Display results
            st.write(f"**Predicted Bias:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            # Show probability distribution
            if use_comprehensive:
                classes = model.classes_
                probabilities = proba
            else:
                classes = classifier.classes_
                probabilities = proba
            
            prob_df = pd.DataFrame({
                'Class': classes,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            st.write("**Probability Distribution:**")
            st.bar_chart(prob_df.set_index('Class'))
            
            # Create visualization if we have embeddings
            if reducer is not None and dataset_embedding is not None:
                if use_comprehensive:
                    # Get the vectorizer from the pipeline
                    vectorizer = model.named_steps['tfidf']
                    vectors = vectorizer.transform([processed_input])
                else:
                    vectors = vectorizer.transform([processed_input])
                
                # Transform user input to UMAP space
                user_embedding = reducer.transform(vectors)
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot existing data
                colors = {'Left': 'blue', 'Center': 'green', 'Right': 'red'}
                for label in df['label'].unique():
                    idx = df['label'] == label
                    ax.scatter(
                        dataset_embedding[idx, 0],
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
                st.info("Visualization not available. Please run the visualization script first.")
        else:
            st.warning("Please enter text to analyze")

if __name__ == "__main__":
    main()