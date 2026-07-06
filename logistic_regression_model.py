import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load dataset
df = pd.read_csv("final_dataset.csv")
# Remove rows with missing target
df = df.dropna(subset=["STATUS_BIN"])
# Drop STATUS column
df = df.drop(columns=["STATUS"])
# Encode all text columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))
# Convert target to integer
df["STATUS_BIN"] = df["STATUS_BIN"].astype(int)
# Features and target
X = df.drop(columns=["STATUS_BIN"])
y = df["STATUS_BIN"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))