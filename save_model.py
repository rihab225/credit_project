import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("final_dataset.csv")

# Remove rows where target is missing
df = df.dropna(subset=["STATUS_BIN"])

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["STATUS_BIN"])
y = df["STATUS_BIN"]

print("Missing values in y:", y.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")