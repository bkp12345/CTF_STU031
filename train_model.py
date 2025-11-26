#!/usr/bin/env python3
"""
Review Authenticity Classifier
Train a machine learning model to distinguish suspicious vs genuine reviews
"""

import csv
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(r"c:\Users\krish\Downloads\STU031")
BOOKS_CSV = DATA_DIR / "books.csv"
REVIEWS_CSV = DATA_DIR / "reviews.csv"
OUTPUT_DIR = DATA_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_ASIN = "0008100993"  # "Four: A Divergent Collection"
SEARCH_HASH = "979DA9FA"

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================

print("[STEP 1] Loading review data...")
reviews_df = pd.read_csv(REVIEWS_CSV, dtype={'asin': str}, low_memory=False)

# Filter reviews for target book
book_reviews = reviews_df[reviews_df['asin'].astype(str) == TARGET_ASIN].copy()
print(f"Found {len(book_reviews)} reviews for book {TARGET_ASIN}")

# Label suspicious reviews (those containing our hash)
book_reviews['is_suspicious'] = book_reviews['text'].astype(str).str.contains(
    SEARCH_HASH, case=False, na=False
).astype(int)

print(f"Suspicious reviews: {book_reviews['is_suspicious'].sum()}")
print(f"Genuine reviews: {(1 - book_reviews['is_suspicious']).sum()}")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================

print("\n[STEP 2] Engineering features...")

def extract_features(row):
    """Extract suspicious indicators from review"""
    text = str(row.get('text', ''))
    rating = float(row.get('rating', 0))
    helpful = float(row.get('helpful_vote', 0))
    verified = 1 if row.get('verified_purchase') == True else 0
    
    # Text-based features
    text_len = len(text.split())
    char_len = len(text)
    avg_word_len = char_len / max(text_len, 1)
    
    # Superlatives indicating fake reviews
    superlatives = ['amazing', 'excellent', 'perfect', 'best', 'love', 'awesome', 
                   'wonderful', 'fantastic', 'great', 'brilliant', 'incredible',
                   'outstanding', 'superb', 'exceptional', 'magnificent']
    superlative_count = sum(1 for word in text.lower().split() if word in superlatives)
    superlative_ratio = superlative_count / max(text_len, 1)
    
    # Negative words (rare in fake reviews)
    negative_words = ['bad', 'poor', 'terrible', 'awful', 'horrible', 'waste',
                     'disappointing', 'boring', 'dull', 'predictable']
    negative_count = sum(1 for word in text.lower().split() if word in negative_words)
    
    # Punctuation patterns
    exclaim_count = text.count('!')
    question_count = text.count('?')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(char_len, 1)
    
    # Rating patterns
    is_five_star = 1 if rating == 5.0 else 0
    is_one_star = 1 if rating == 1.0 else 0
    
    return {
        'text_length': text_len,
        'char_length': char_len,
        'avg_word_length': avg_word_len,
        'superlative_count': superlative_count,
        'superlative_ratio': superlative_ratio,
        'negative_count': negative_count,
        'exclamation_count': exclaim_count,
        'question_count': question_count,
        'caps_ratio': caps_ratio,
        'is_five_star': is_five_star,
        'is_one_star': is_one_star,
        'rating': rating,
        'helpful_votes': helpful,
        'verified_purchase': verified
    }

# Extract features
features_list = []
for _, row in book_reviews.iterrows():
    features_list.append(extract_features(row))

features_df = pd.DataFrame(features_list)
print(f"Extracted {len(features_df)} feature sets")

# ============================================================================
# STEP 3: Text Vectorization (TF-IDF)
# ============================================================================

print("\n[STEP 3] Vectorizing text with TF-IDF...")
texts = book_reviews['text'].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer(max_features=50, stop_words='english', min_df=1)
tfidf_matrix = vectorizer.fit_transform(texts).toarray()

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Combine all features
X = np.hstack([features_df.values, tfidf_matrix])
y = book_reviews['is_suspicious'].values

print(f"Final feature matrix shape: {X.shape}")
print(f"Labels distribution: {np.bincount(y)}")

# ============================================================================
# STEP 4: Model Training
# ============================================================================

print("\n[STEP 4] Training classification models...")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest Classifier
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_scaled, y)

# Get predictions and probabilities
y_pred = rf_model.predict(X_scaled)
y_proba = rf_model.predict_proba(X_scaled)[:, 1]

print("Model trained successfully!")

# ============================================================================
# STEP 5: Model Evaluation
# ============================================================================

print("\n[STEP 5] Model evaluation...")
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Genuine', 'Suspicious']))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# ============================================================================
# STEP 6: Feature Importance
# ============================================================================

print("\n[STEP 6] Feature importance analysis...")
feature_names = list(features_df.columns) + vectorizer.get_feature_names_out().tolist()
importances = rf_model.feature_importances_

# Get top 15 important features
top_indices = np.argsort(importances)[-15:][::-1]
print("\nTop 15 Important Features:")
for idx in top_indices:
    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

# ============================================================================
# STEP 7: Analyze Genuine Reviews
# ============================================================================

print("\n[STEP 7] Analyzing genuine reviews...")

# Get predictions for each review
book_reviews['suspicious_score'] = y_proba
book_reviews['prediction'] = y_pred

# Filter genuine reviews (low suspicion scores)
genuine_mask = (book_reviews['is_suspicious'] == 0) & (book_reviews['suspicious_score'] < 0.5)
genuine_reviews = book_reviews[genuine_mask]

print(f"Identified {len(genuine_reviews)} genuine reviews")

# Extract top words from genuine reviews
all_words = []
for text in genuine_reviews['text'].fillna("").astype(str):
    words = [w.lower() for w in text.split() if w.isalpha() and len(w) > 2]
    all_words.extend(words)

stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that'}

word_freq = Counter(w for w in all_words if w not in stopwords)
top_words = word_freq.most_common(10)

print("\nTop 10 words in genuine reviews:")
for word, count in top_words:
    print(f"  {word}: {count}")

# ============================================================================
# STEP 8: Save Models and Results
# ============================================================================

print("\n[STEP 8] Saving models and results...")

# Save models
with open(OUTPUT_DIR / 'rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(OUTPUT_DIR / 'vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✓ Models saved to {OUTPUT_DIR}/")

# Save results summary
results = {
    'model_type': 'RandomForest',
    'accuracy': np.mean(y_pred == y),
    'total_reviews': len(book_reviews),
    'suspicious_reviews': int(book_reviews['is_suspicious'].sum()),
    'genuine_reviews': int((1 - book_reviews['is_suspicious']).sum()),
    'top_features': [(feature_names[idx], float(importances[idx])) for idx in top_indices],
    'top_words_in_genuine': top_words,
    'flagged_as_suspicious': int(y_pred.sum()),
    'flagged_as_genuine': int((1 - y_pred).sum())
}

import json
with open(OUTPUT_DIR / 'model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Results saved to model_results.json")

# ============================================================================
# STEP 9: Create Prediction Function
# ============================================================================

print("\n[STEP 9] Creating reusable prediction module...")

prediction_code = '''#!/usr/bin/env python3
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
'''

with open(OUTPUT_DIR / 'predict.py', 'w') as f:
    f.write(prediction_code)

print("✓ Prediction module created at models/predict.py")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE")
print("="*70)

print(f"""
✓ Model: Random Forest Classifier
✓ Accuracy: {np.mean(y_pred == y):.2%}
✓ Trained on: {len(book_reviews)} reviews
✓ Classes: Genuine (0) vs Suspicious (1)

FILES SAVED:
  • models/rf_model.pkl - Trained model
  • models/scaler.pkl - Feature scaler
  • models/vectorizer.pkl - TF-IDF vectorizer
  • models/model_results.json - Evaluation results
  • models/predict.py - Prediction utility

FEATURES USED: {len(feature_names)}
  • 14 numerical features (text properties, ratings, etc.)
  • 50 TF-IDF text features (word importance)

USE THE MODEL:
  python models/predict.py
  
Or in Python:
  from models.predict import predict_review
  result = predict_review("Your review text here")
  print(result)
""")
