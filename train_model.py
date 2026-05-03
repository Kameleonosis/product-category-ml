import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. Učitavanje podataka
# =========================
df = pd.read_csv("products.csv")

# 🔥 VAŽNO: uklanja razmake u nazivima kolona
df.columns = df.columns.str.strip()

# =========================
# 2. Čišćenje podataka
# =========================
df.dropna(inplace=True)

# Provera kolona (debug)
print("Kolone:", df.columns.tolist())

# =========================
# 3. Definisanje X i y
# =========================
X = df["Product Title"]
y = df["Category Label"]

# =========================
# 4. Podela na train/test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. TF-IDF vektorizacija
# =========================
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 6. Treniranje modela
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# =========================
# 7. Evaluacija (BONUS za ocenu)
# =========================
y_pred = model.predict(X_test_vec)

print("\n--- Rezultati modela ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 8. Čuvanje modela
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("\nModel sacuvan kao model.pkl")