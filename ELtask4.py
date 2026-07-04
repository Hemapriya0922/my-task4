# Step 1: Import Libraries and Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# Load the dataset
df = pd.read_csv(r"C:\Users\hemap\Downloads\data.csv")

# Remove unnecessary columns
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Convert target variable into numeric
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print("Binary Classification Dataset Selected")
print(df["diagnosis"].value_counts())


# ===========================================
# Step 2: Train/Test Split and Standardization
# ===========================================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)


# ===========================================
# Step 3: Fit Logistic Regression Model
# ===========================================

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ===========================================
# Step 4: Evaluation
# ===========================================

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ===========================================
# Step 5: Threshold Tuning and Sigmoid Function
# ===========================================

# Change threshold from default 0.5 to 0.3
threshold = 0.3

y_pred_new = (y_prob >= threshold).astype(int)

print("\nResults After Threshold =", threshold)
print(confusion_matrix(y_test, y_pred_new))
print(classification_report(y_test, y_pred_new))

# Sigmoid Function
x = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-x))

plt.figure(figsize=(6,4))
plt.plot(x, sigmoid)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Probability")
plt.grid(True)
plt.show()
