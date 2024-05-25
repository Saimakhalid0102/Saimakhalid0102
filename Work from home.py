import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Generate Synthetic Data
np.random.seed(0)

# Define possible values for each feature
regions = ['North', 'South', 'East', 'West']
product_types = ['Type A', 'Type B', 'Type C']
purchase_behaviors = ['High', 'Medium', 'Low']
responses = ['Yes', 'No']

# Number of samples
n_samples = 1000

# Randomly generate data
data = {
    'Region': np.random.choice(regions, n_samples),
    'Product Type': np.random.choice(product_types, n_samples),
    'Purchase Behavior': np.random.choice(purchase_behaviors, n_samples),
    'Response': np.random.choice(responses, n_samples, p=[0.3, 0.7])  # Assume 30% positive response rate
}

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Prepare Data for Model
X = df[['Region', 'Product Type', 'Purchase Behavior']]
y = df['Response']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Show example predictions
example_predictions = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}).head(10)
print(example_predictions)
