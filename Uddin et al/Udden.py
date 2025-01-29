import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Input
from tensorflow.keras.utils import to_categorical

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# Load datasets
train_df = pd.read_csv('Uddin et al/train_loan_pred.csv')
test_df = pd.read_csv('Uddin et al/test_loan_pred.csv')

# Data preprocessing
def preprocess_data(df, is_train=True):
    df = df.copy()

    # Fill missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

    # Handle '3+' in Dependents column
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    # Encode categorical variables
    label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Normalize numerical variables
    scaler = StandardScaler()
    df[['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']] = scaler.fit_transform(df[['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']])
    return df

# Preprocess train and test data
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df, is_train=False)

# Separate features and target
X_train = train_df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y_train = train_df['Loan_Status'].map({'Y': 1, 'N': 0})

X_test = test_df.drop(['Loan_ID'], axis=1)

# Balance data using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train individual models and evaluate
def evaluate_models():
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNeighbors': KNeighborsClassifier(),
        'GaussianNB': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_train_balanced)

        accuracy = accuracy_score(y_train_balanced, y_pred)
        precision = precision_score(y_train_balanced, y_pred)
        recall = recall_score(y_train_balanced, y_pred)
        f1 = f1_score(y_train_balanced, y_pred)

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    results_df = pd.DataFrame(results)
    print("\nModel Performance Metrics:\n")
    print(results_df)
    results_df.to_csv('Uddin et al/model_performance_metrics.csv', index=False)
    print("\nMetrics saved to 'Uddin et al/model_performance_metrics.csv'.")

# Evaluate models
evaluate_models()
