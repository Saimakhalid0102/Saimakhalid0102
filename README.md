import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'path_to_your_dataset/diabetes (1).csv'
df_nb = pd.read_csv(file_path)

# Separate features and target variable
X_nb = df_nb.drop('Outcome', axis=1)
y_nb = df_nb['Outcome']

# Split the dataset into training and testing sets
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)

# Standardize the features
scaler_nb = StandardScaler()
X_train_nb = scaler_nb.fit_transform(X_train_nb)
X_test_nb = scaler_nb.transform(X_test_nb)

# Train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_nb, y_train_nb)

# Make predictions
y_pred_nb = nb_classifier.predict(X_test_nb)

# Calculate accuracy and precision
accuracy_nb = accuracy_score(y_test_nb, y_pred_nb)
precision_nb = precision_score(y_test_nb, y_pred_nb)

print(f"Accuracy: {accuracy_nb}")
print(f"Precision: {precision_nb}")

# Generate the ROC curve
fpr_nb, tpr_nb, _nb = roc_curve(y_test_nb, nb_classifier.predict_proba(X_test_nb)[:, 1])

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_nb, tpr_nb, color='blue', label='Naive Bayes')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Generate the confusion matrix
conf_matrix_nb = confusion_matrix(y_test_nb, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_nb)
disp_nb.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show() 
