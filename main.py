import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ransomdata=pd.read_csv(r"C:/Users/Juma/Desktop/attack/data_file.csv")
print(ransomdata.head())
print(ransomdata .isnull().sum())
print(ransomdata.dtypes)

ransomdata.drop(columns=["FileName", "md5Hash","BitcoinAddresses"], inplace=True)


correlation_ch=ransomdata.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_ch, cmap="coolwarm", annot=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


benign_correlation = correlation_ch["Benign"].sort_values(ascending=False)
print(benign_correlation)


check_balance =ransomdata["Benign"].value_counts(normalize=True) * 100
print(check_balance)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
ransomdata_scaled=scaler.fit_transform(ransomdata)


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Split data into features (X) and target (y)
X = ransomdata[['DebugRVA',"Machine","MajorOSVersion","MajorLinkerVersion","DllCharacteristics","IatVRA","MajorImageVersion"]]
y = ransomdata["Benign"]

# Apply SMOTE to generate synthetic samples
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/test split after balancing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)


# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))



train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")


cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Malicious (0)", "Benign (1)"], yticklabels=["Malicious (0)", "Benign (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()


from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=5, scoring="accuracy")

# Print mean and standard deviation of cross-validation accuracy
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


import joblib
joblib.dump(rf_model, "rf_model.pkl")


