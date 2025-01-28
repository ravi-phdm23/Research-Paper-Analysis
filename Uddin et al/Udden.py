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
from sklearn.metrics import accuracy_score, classification_report
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

# Convert target to categorical for neural networks
y_train_categorical = to_categorical(y_train_balanced)

# Dense Neural Network
dense_model = Sequential([
    Input(shape=(X_train_balanced.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
dense_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dense_model.fit(X_train_balanced, y_train_categorical, epochs=10, batch_size=32, verbose=0)

# LSTM Neural Network
lstm_model = Sequential([
    Input(shape=(1, X_train_balanced.shape[1])),
    LSTM(64, activation='tanh', return_sequences=False),
    Dense(2, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_balanced.values.reshape(-1, 1, X_train_balanced.shape[1]), y_train_categorical, epochs=10, batch_size=32, verbose=0)

# Recurrent Neural Network (SimpleRNN)
rnn_model = Sequential([
    Input(shape=(1, X_train_balanced.shape[1])),
    SimpleRNN(64, activation='tanh', return_sequences=False),
    Dense(2, activation='softmax')
])
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train_balanced.values.reshape(-1, 1, X_train_balanced.shape[1]), y_train_categorical, epochs=10, batch_size=32, verbose=0)

# Train individual models
lr = LogisticRegression(random_state=42, max_iter=500)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
et = ExtraTreesClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
knn = KNeighborsClassifier()
gnb = GaussianNB()
adb = AdaBoostClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

lr.fit(X_train_balanced, y_train_balanced)
dt.fit(X_train_balanced, y_train_balanced)
rf.fit(X_train_balanced, y_train_balanced)
et.fit(X_train_balanced, y_train_balanced)
svm.fit(X_train_balanced, y_train_balanced)
knn.fit(X_train_balanced, y_train_balanced)
gnb.fit(X_train_balanced, y_train_balanced)
adb.fit(X_train_balanced, y_train_balanced)
gb.fit(X_train_balanced, y_train_balanced)

# Combine models into a voting ensemble
voting_clf = VotingClassifier(estimators=[
    ('lr', lr), ('dt', dt), ('rf', rf), ('et', et), ('svm', svm), ('knn', knn), ('gnb', gnb), ('adb', adb), ('gb', gb)
], voting='soft')

voting_clf.fit(X_train_balanced, y_train_balanced)

# Predict and evaluate
train_preds = voting_clf.predict(X_train_balanced)
train_accuracy = accuracy_score(y_train_balanced, train_preds)
print("Training Accuracy:", train_accuracy)

# Predictions on test data
nn_preds = np.argmax(dense_model.predict(X_test), axis=1)
test_preds = voting_clf.predict(X_test)
final_preds = (nn_preds + test_preds) // 2  # Combine neural networks and ensemble model

test_df['Loan_Status'] = pd.Series(final_preds).map({1: 'Y', 0: 'N'})

# Save results
test_df[['Loan_ID', 'Loan_Status']].to_csv('Uddin et al/test_predictions.csv', index=False)
print("Test predictions saved to 'Uddin et al/test_predictions.csv'.")
