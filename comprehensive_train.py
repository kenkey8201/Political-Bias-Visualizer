# comprehensive_train.py
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Calculate class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# Create a pipeline with TF-IDF and SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,  # Increased features
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )),
    ('scaler', StandardScaler(with_mean=False)),
    ('svm', SVC(
        kernel='linear',
        probability=True,
        class_weight=class_weight_dict,
        C=1.0,  # Regularization parameter
        random_state=42
    ))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(pipeline, 'comprehensive_classifier.pkl')
print("Model saved as comprehensive_classifier.pkl")