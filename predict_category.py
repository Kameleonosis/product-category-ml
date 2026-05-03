import pickle

# Ucitavanje modela
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

print("Unesi naziv proizvoda (exit za izlaz)")

while True:
    text = input("Naziv: ")

    if text.lower() == "exit":
        break

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    print("Predikcija:", pred)