import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix
from tpot import TPOTClassifier

# Step 1: Load Data
data = pd.read_csv('features.csv')  # Replace with your dataset file name

# Step 2: Drop 'report_date' and separate the target variable
X = data.drop(columns=['report_date', 'risk_class'])  # Drop unnecessary columns
y = data['risk_class']  # Target column

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert 'High', 'Low', etc., to integers

# Print the label encoding mapping
print("Label Encoding Mapping:")
for idx, label in enumerate(label_encoder.classes_):
    print(f"{label}: {idx}")

# Step 3: Preprocessing - Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Feature Selection using Random Forest
feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
feature_selector.fit(X_scaled, y)

# Select top features based on importance
important_features = np.where(feature_selector.feature_importances_ > 0.01)[0]
X_selected = X_scaled[:, important_features]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

# Step 6: Model Training and Evaluation

# Logistic Regression (Baseline)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
f1_log = f1_score(y_test, y_pred_log, average='weighted')

# Random Forest
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# XGBoost
xgb = XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=7, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')

# TPOT (AutoML)
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=3, scoring='f1_weighted', random_state=42)
tpot.fit(X_train, y_train)
f1_tpot = f1_score(y_test, tpot.predict(X_test), average='weighted')

# Step 7: Evaluation and Comparison
print(f"\nF1 Scores:")
print(f"Logistic Regression: {f1_log:.4f}")
print(f"Random Forest: {f1_rf:.4f}")
print(f"XGBoost: {f1_xgb:.4f}")
print(f"TPOT: {f1_tpot:.4f}")

# Confusion Matrix Example
print("\nConfusion Matrix (XGBoost):")
print(confusion_matrix(y_test, y_pred_xgb))

# Optional: Save selected features for interpretability
selected_feature_names = data.columns[1:][important_features]
selected_feature_names.to_csv('selected_features.csv', index=False)
