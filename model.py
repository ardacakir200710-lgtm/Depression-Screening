import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


# Load the dataset

df = pd.read_csv("clear_data.csv")

# Define features and target
X = df.drop(columns="Risk_Derecesi")
y = df["Risk_Derecesi"]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the logistic regression model
model = LogisticRegression(max_iter=2000, multi_class="multinomial")
model.fit(X_train, y_train)

# Measure model accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model for use in the API
joblib.dump(model, "risk_model.joblib")
