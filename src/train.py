import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import preprocess
from model import get_model

df = pd.read_csv("/Users/anshulraghu/Desktop/Spring2026/EGN 6217/Project/data/Personal_Finance_Dataset.csv")
df = preprocess(df)

X = df[['amount']]
y = df['category_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = get_model()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Accuracy: {acc}")

# Save metric
with open("../results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}")