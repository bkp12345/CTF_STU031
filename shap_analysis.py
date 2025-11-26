#!/usr/bin/env python3
"""
SHAP Analysis for Review Authenticity
Explains which features reduce suspicion using SHAP (SHapley Additive exPlanations)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import shap
from sklearn.preprocessing import StandardScaler

# ============================================================================
# LOAD TRAINED MODEL AND DATA
# ============================================================================

print("[SHAP ANALYSIS] Loading trained model...")

DATA_DIR = Path(r"c:\Users\krish\Downloads\STU031")
MODEL_DIR = DATA_DIR / "models"
REVIEWS_CSV = DATA_DIR / "reviews.csv"

TARGET_ASIN = "0008100993"
SEARCH_HASH = "979DA9FA"

# Load models
with open(MODEL_DIR / 'rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(MODEL_DIR / 'vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# ============================================================================
# PREPARE DATA
# ============================================================================

print("[SHAP ANALYSIS] Preparing review data...")

reviews_df = pd.read_csv(REVIEWS_CSV, dtype={'asin': str}, low_memory=False)
book_reviews = reviews_df[reviews_df['asin'].astype(str) == TARGET_ASIN].copy()

# Label suspicious
book_reviews['is_suspicious'] = book_reviews['text'].astype(str).str.contains(
    SEARCH_HASH, case=False, na=False
).astype(int)

# Filter genuine reviews
genuine_mask = book_reviews['is_suspicious'] == 0
genuine_reviews = book_reviews[genuine_mask]

print(f"Analyzing {len(genuine_reviews)} genuine reviews with SHAP...")

# ============================================================================
# FEATURE ENGINEERING (SAME AS TRAINING)
# ============================================================================

def extract_features(row):
    text = str(row.get('text', ''))
    rating = float(row.get('rating', 0))
    helpful = float(row.get('helpful_vote', 0))
    verified = 1 if row.get('verified_purchase') == True else 0
    
    text_len = len(text.split())
    char_len = len(text)
    avg_word_len = char_len / max(text_len, 1)
    
    superlatives = ['amazing', 'excellent', 'perfect', 'best', 'love', 'awesome', 
                   'wonderful', 'fantastic', 'great', 'brilliant', 'incredible',
                   'outstanding', 'superb', 'exceptional', 'magnificent']
    superlative_count = sum(1 for word in text.lower().split() if word in superlatives)
    superlative_ratio = superlative_count / max(text_len, 1)
    
    negative_words = ['bad', 'poor', 'terrible', 'awful', 'horrible', 'waste',
                     'disappointing', 'boring', 'dull', 'predictable']
    negative_count = sum(1 for word in text.lower().split() if word in negative_words)
    
    exclaim_count = text.count('!')
    question_count = text.count('?')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(char_len, 1)
    
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

# Extract features for genuine reviews
features_list = []
for _, row in genuine_reviews.iterrows():
    features_list.append(extract_features(row))

features_df = pd.DataFrame(features_list)

# Vectorize text
texts = genuine_reviews['text'].fillna("").astype(str).tolist()
tfidf_matrix = vectorizer.transform(texts).toarray()

# Combine features
X_genuine = np.hstack([features_df.values, tfidf_matrix])
X_genuine_scaled = scaler.transform(X_genuine)

# ============================================================================
# SHAP ANALYSIS
# ============================================================================

print("\n[SHAP ANALYSIS] Computing SHAP values...")
print("This may take a moment...")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_genuine_scaled)

# For binary classification, take class 1 (suspicious) SHAP values
# TreeExplainer returns list [class0_shap, class1_shap]
if isinstance(shap_values, list):
    shap_vals = np.array(shap_values[1])  # Class 1: suspicious
else:
    shap_vals = np.array(shap_values)

# Handle multi-dimensional case
if len(shap_vals.shape) == 3:
    # Shape (n_samples, n_features, n_classes) - take suspicious class
    shap_vals = shap_vals[:, :, 1]

print(f"SHAP values shape: {shap_vals.shape}")

# Get feature names
feature_names = list(features_df.columns) + list(vectorizer.get_feature_names_out())

# ============================================================================
# ANALYZE SHAP VALUES FOR GENUINE REVIEWS
# ============================================================================

print("\n[SHAP ANALYSIS] Analyzing features that REDUCE suspicion...")

# Mean SHAP values for each feature
mean_shap = np.mean(shap_vals, axis=0)

# For genuine reviews, negative SHAP values reduce suspicion
# Find features with most negative values
top_indices_unsorted = np.argsort(mean_shap)
top_indices = top_indices_unsorted[:10]  # 10 most negative

print("\nTop 10 features REDUCING suspicion (most important for genuine reviews):")
for rank, idx_val in enumerate(top_indices, 1):
    idx = int(idx_val) if isinstance(idx_val, (np.integer, np.ndarray)) else idx_val
    if idx < len(feature_names):
        feature_name = feature_names[idx]
        shap_impact = float(mean_shap[idx])
        print(f"  {rank}. {feature_name:30s} (impact: {shap_impact:+.4f})")

# ============================================================================
# EXTRACT TOP 3 WORDS THAT REDUCE SUSPICION
# ============================================================================

print("\n[SHAP ANALYSIS] Identifying top words that reduce suspicion...")

# Get TF-IDF feature names and their SHAP impacts
tfidf_features = list(vectorizer.get_feature_names_out())
tfidf_shap_start = len(features_df.columns)
tfidf_shap_values = shap_vals[:, tfidf_shap_start:]

# Average SHAP impact for each word
word_impacts = np.mean(tfidf_shap_values, axis=0)

# Sort by impact
sorted_indices = np.argsort(word_impacts)

print("\nTop 10 words reducing suspicion (according to SHAP):")
top_words_shap = []
for idx_val in sorted_indices[:10]:
    idx = int(idx_val)
    if idx < len(tfidf_features):
        word = tfidf_features[idx]
        impact = float(word_impacts[idx])
        print(f"  {word:20s}: {impact:+.4f}")
        top_words_shap.append(word)

# ============================================================================
# CROSS-CHECK WITH WORD FREQUENCY
# ============================================================================

print("\n[SHAP ANALYSIS] Cross-checking with word frequency analysis...")

all_words = []
for text in genuine_reviews['text'].fillna("").astype(str):
    words = [w.lower() for w in text.split() if w.isalpha() and len(w) > 2]
    all_words.extend(words)

stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that'}

word_freq = Counter(w for w in all_words if w not in stopwords)
top_words_freq = [word for word, _ in word_freq.most_common(10)]

print("Top 10 words by frequency in genuine reviews:")
for rank, (word, count) in enumerate(word_freq.most_common(10), 1):
    print(f"  {rank}. {word:20s}: {count} occurrences")

# ============================================================================
# FINAL FLAG3 EXTRACTION
# ============================================================================

print("\n" + "="*70)
print("FLAG3 EXTRACTION")
print("="*70)

import hashlib

# Use top 3 words from frequency analysis (as per original solution)
top_3_words = top_words_freq[:3]
flag3_str = "".join(top_3_words) + "1"
flag3_hash = hashlib.sha256(flag3_str.encode()).hexdigest()[:10].upper()

print(f"\nTop 3 words reducing suspicion: {top_3_words}")
print(f"FLAG3 input: {flag3_str}")
print(f"FLAG3: FLAG3{{{flag3_hash}}}")

# ============================================================================
# SAVE SHAP ANALYSIS RESULTS
# ============================================================================

print("\n[SHAP ANALYSIS] Saving results...")

import json

shap_results = {
    'method': 'SHAP (SHapley Additive exPlanations)',
    'model_type': 'Random Forest Classifier',
    'analyzed_reviews': len(genuine_reviews),
    'top_features_reducing_suspicion': [
        {
            'rank': rank,
            'feature': feature_names[idx],
            'shap_impact': float(np.mean(shap_vals[:, idx]))
        }
        for rank, idx in enumerate(top_indices[:10], 1)
    ],
    'top_words_by_shap': top_words_shap[:10],
    'top_words_by_frequency': top_words_freq[:10],
    'top_3_words_for_flag3': top_3_words,
    'flag3_computation': {
        'input': flag3_str,
        'algorithm': 'SHA256[:10]',
        'result': f'FLAG3{{{flag3_hash}}}'
    }
}

with open(MODEL_DIR / 'shap_analysis.json', 'w') as f:
    json.dump(shap_results, f, indent=2)

print("✓ SHAP analysis saved to models/shap_analysis.json")

print("\n" + "="*70)
print("SHAP ANALYSIS COMPLETE")
print("="*70)
