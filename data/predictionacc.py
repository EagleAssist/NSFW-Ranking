from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd

# Load the data
DATA_FILE_NAME = "clean_data.csv"
data = pd.read_csv(DATA_FILE_NAME)
texts = data["text"].astype(str)
y = data["is_offensive"]

# Load the pre-trained vectorizer and model
vectorizer = load("vectorizer.joblib")
calibrated_classifier_cv = load("model.joblib")

# Transform the text data using the loaded vectorizer
X = vectorizer.transform(texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the calibrated classifier on the training data
calibrated_classifier_cv.fit(X_train, y_train)

# Predict the labels on the test set
y_pred = calibrated_classifier_cv.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")