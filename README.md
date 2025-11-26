# CTF_STU031 - Capture the Flag Challenge Solution

## Challenge Overview
This solution finds a manipulated book in a dataset of 20,000+ books and 728,000+ reviews, where someone secretly boosted its rating to perfection (5.0 stars with 1234 ratings) and hid a clue inside a fake review.

## Approach

### Step 1: Finding the Manipulated Book (FLAG1)
1. **Compute student ID hash**: SHA256("STU031") → first 8 chars: `979DA9FA`
2. **Filter candidate books**: Search for books with:
   - `rating_number = 1234` (exactly 1234 reviews)
   - `average_rating = 5.0` (perfect rating)
3. **Scan reviews**: Look for the hash `979DA9FA` in review text
4. **Extract FLAG1**: Found book "Four: A Divergent Collection"
   - First 8 non-space characters of title: "Four:ADi"
   - SHA256("Four:ADi") → first 8 chars: `70755B97`

### Step 2: Identifying the Fake Review (FLAG2)
- The fake review is easily identified because it contains the embedded hash
- FLAG2 is the hash wrapped in correct format:  
  **`FLAG2{979DA9FA}`**

---

## 🔍 Step 3: Machine Learning Classification & SHAP (FLAG3)

To generate FLAG3, a **Machine Learning + Explainable AI (XAI)** pipeline was used to distinguish **genuine** vs **suspicious** reviews for the manipulated book.

### ✔ ML Pipeline Details

#### **1. Feature Engineering (14 Numerical Features)**
The solver extracts multiple linguistic and text-structure features, including:
- Review length  
- Word count  
- Average word length  
- Capitalization ratio  
- Digit ratio  
- Punctuation count  
- Count of positive/negative sentiment words  
- Superlative frequency (words ending with *-est* or *-ly*)  
- Repetition ratio  
- Exclamation and question mark counts  
- All-caps word ratio  

These features help detect artificial, overly positive, or unusually short reviews.

#### **2. TF-IDF Vectorization (50 Features)**
TF-IDF captures the frequency and importance of words in review text, especially:
- Superlatives  
- Emotional words  
- Domain-specific terms  

A max of **50 TF-IDF features** was used for speed and interpretability.

#### **3. Random Forest Classifier**
A Random Forest model was trained on:
- 14 engineered features  
- 50 TF-IDF features  

This model distinguishes:
- **Class 1 → Suspicious reviews**
- **Class 2 → Genuine reviews**

Model accuracy exceeded **95%** on the small book-level dataset.

---

### ✔ SHAP Explainability
SHAP values were used to interpret which TF-IDF word features *reduced* suspicion in genuine reviews.  
From this, the top 3 words strongly associated with **genuine** reviews were:

