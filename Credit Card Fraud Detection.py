# Credit Card Fraud Detection


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print(df.head())


fraud_count = df['Class'].value_counts()
print("\nClass Distribution:\n", fraud_count)

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Imbalance (0 = Legit, 1 = Fraud)")
plt.show()


X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_smote).value_counts())


rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_smote, y_smote)


y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


custom_threshold = 0.3
y_custom = (y_prob >= custom_threshold).astype(int)

print("\nClassification Report at Threshold = 0.3\n")
print(classification_report(y_test, y_custom))

"""
Business Discussion:
- Lower threshold increases Recall (catch more frauds)
- Higher threshold increases Precision (fewer false alerts)
- Banks prefer higher Recall to minimize financial loss
"""
