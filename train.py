from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from vectorize import vectorize_text
import joblib

def train_classifier():
    # Load and prepare data
    df = pd.read_csv('dataset/dataset.csv')
    X, vectorizer = vectorize_text(df['text'])
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train SVM classifier
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(clf, 'classifier.pkl')
    return clf, vectorizer

if __name__ == "__main__":
    train_classifier()