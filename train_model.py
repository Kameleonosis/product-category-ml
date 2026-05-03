import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ucitavanje
df = pd.read_csv("products.csv")

# Ciscenje
df.dropna(inplace=True)

X = df["Product Title"]
y = df["Category Label"]

# Podela
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Cuvanje
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Model sacuvan kao model.pkl")