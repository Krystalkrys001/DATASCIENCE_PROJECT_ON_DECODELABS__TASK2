# =========================================
# PROJECT 2: Supervised Learning, Fraud Detection Pipeline
# DecodeLabs Data Science Industrial Training, 2026 Batch
# =========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# =========================================
# STEP 1: Load and inspect the data
# =========================================
df = pd.read_csv("creditcard.csv")
print(df.shape)                     # (284807, 31)
print(df.isnull().sum().sum())      # 0, no missing values in this dataset
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True) * 100)
# Fraud rate: 0.17%. This imbalance is the entire reason this project exists.

# =========================================
# STEP 2: Prove why "accuracy" is a trap on this data
# =========================================
lazy_predictions = [0] * len(df)
lazy_accuracy = (lazy_predictions == df["Class"]).mean()
print(f"A model that predicts 'legitimate' for everything: {lazy_accuracy*100:.2f}% accuracy")
print("Fraud caught by that lazy model: 0 out of", df["Class"].sum())
# 99.83% accuracy, 0 fraud caught. Accuracy alone is meaningless here.

# =========================================
# STEP 3: Train/test split (stratified, BEFORE any resampling)
# =========================================
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y keeps the same 0.17% fraud ratio in both train and test

# =========================================
# STEP 4a: Logistic Regression pipeline
# Scale FIRST, then SMOTE (SMOTE uses distance, unscaled "Amount"
# would dominate the nearest-neighbor calculation otherwise)
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)      # fit only on train
X_test_scaled = scaler.transform(X_test)             # test only ever transformed, never fit

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_balanced, y_train_balanced)
print("Logistic Regression trained.")

y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n--- Logistic Regression: Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred_lr))
print("\n--- Logistic Regression: Classification Report ---")
print(classification_report(y_test, y_pred_lr, digits=4))
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_proba_lr))

# =========================================
# STEP 4b: Random Forest pipeline
# No scaling needed, tree-based models split ordinally, immune to scale.
# SMOTE applied directly to raw (unscaled) training data.
# =========================================
smote_rf = SMOTE(random_state=42)
X_train_balanced_rf, y_train_balanced_rf = smote_rf.fit_resample(X_train, y_train)

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=1,          # single core, more reliable on shared/hosted notebooks
    n_estimators=30,   # fewer trees, faster result without meaningfully hurting accuracy
    max_depth=12        # limits tree complexity, speeds up training further
)
rf.fit(X_train_balanced_rf, y_train_balanced_rf)
print("Random Forest trained.")

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest: Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred_rf))
print("\n--- Random Forest: Classification Report ---")
print(classification_report(y_test, y_pred_rf, digits=4))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

# =========================================
# RESULTS SUMMARY
# =========================================
# Logistic Regression : Recall 91.84% | Precision  5.78% | ROC-AUC 97.08%
# Random Forest        : Recall 85.71% | Precision 62.69% | ROC-AUC 97.45%
#
# DECISION: Random Forest is the better model for this use case.
# It trades a small amount of recall (6 fewer fraud cases caught out of 98)
# for a massive gain in precision, cutting false alarms from 1,467 down to 50.
# In production, that difference is what determines whether a fraud team
# trusts and acts on the model's alerts, or drowns in false positives
# and starts ignoring them.
