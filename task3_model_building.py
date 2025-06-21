import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the data)  # Update path if needed
df = pd.read_csv("C:/Users/vathu/Downloads/final_cleaned_data_large.csv")
# Encode Gender: Male = 0, Female = 1
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# One-hot encode 'City'
df = pd.get_dummies(df, columns=["City"], drop_first=True)

# Features & Target
X = df.drop(columns=["Gender", "Name"])
y = df["Gender"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save output to model_output.txt
with open("model_output.txt", "w", encoding="utf-8") as f:
    f.write("==================== Model Training Summary ====================\n")
    f.write(" Data loaded from: final_cleaned_data_large.csv\n")
    f.write(f" Total records: {len(df)}\n")
    f.write(f" Features used: {list(X.columns)}\n\n")

    f.write("==================== Model Evaluation ====================\n")
    f.write(f"🎯 Accuracy: {accuracy:.2f}\n\n")
    f.write("📋 Classification Report:\n")
    f.write(report)
    f.write("\n")
    f.write("==================== Notes ====================\n")
    f.write("✅ Gender was encoded as: Male = 0, Female = 1\n")
    f.write("✅ City was one-hot encoded\n")
    f.write("✅ Model used: LogisticRegression\n")
    f.write("=================================================\n")

print("✅ Model output saved to 'model_output.txt'")
