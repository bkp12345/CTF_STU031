#!/usr/bin/env python3
"""
Predict review authenticity using trained model
"""

import pickle
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent

def predict_review(text, rating=5.0, helpful_votes=0, verified=True):
    """Predict if a review is suspicious (0=genuine, 1=suspicious)"""
    
    # Load models
    with open(MODEL_DIR / 'rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / 'vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Extract features (same as training)
    text_len = len(text.split())
    char_len = len(text)
    avg_word_len = char_len / max(text_len, 1)
    
    superlatives = ['amazing', 'excellent', 'perfect', 'best', 'love', 'awesome',
                   'wonderful', 'fantastic', 'great', 'brilliant']
    superlative_count = sum(1 for word in text.lower().split() if word in superlatives)
    superlative_ratio = superlative_count / max(text_len, 1)
    
    negative_words = ['bad', 'poor', 'terrible', 'awful', 'horrible']
    negative_count = sum(1 for word in text.lower().split() if word in negative_words)
    
    numeric_features = np.array([
        text_len, char_len, avg_word_len, superlative_count, superlative_ratio,
        negative_count, text.count('!'), text.count('?'),
        sum(1 for c in text if c.isupper()) / max(char_len, 1),
        1 if rating == 5.0 else 0, 0, rating, helpful_votes, verified
    ]).reshape(1, -1)
    
    # Vectorize text
    tfidf = vectorizer.transform([text]).toarray()
    
    # Combine features
    X = np.hstack([numeric_features, tfidf])
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0, 1]
    
    return {
        'prediction': 'SUSPICIOUS' if prediction == 1 else 'GENUINE',
        'probability': float(probability),
        'confidence': max(model.predict_proba(X_scaled)[0])
    }

if __name__ == '__main__':
    # Test example
    test_review = "Perfect amazing experience 979DA9FA"
    result = predict_review(test_review)
    print(f"Review: {test_review}")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability (suspicious): {result['probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
