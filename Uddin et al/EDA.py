import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_df = pd.read_csv('Uddin et al/train_loan_pred.csv')
test_df = pd.read_csv('Uddin et al/test_loan_pred.csv')

# Function to perform exploratory data analysis (EDA)
def perform_eda(df, dataset_name):
    print(f"\n--- EDA for {dataset_name} ---\n")
    
    # Dataset Overview
    print(f"Shape of the dataset: {df.shape}\n")
    print("First 5 rows:\n", df.head())
    
    # Check for missing values
    print("\nMissing values:\n", df.isnull().sum())

    # Summary statistics
    print("\nSummary statistics:\n", df.describe())

    # Target variable distribution
    if 'Loan_Status' in df.columns:
        print("\nLoan Status Distribution:\n")
        sns.countplot(data=df, x='Loan_Status', palette='Set2')
        plt.title(f"Loan Status Distribution in {dataset_name}")
        plt.show()

    # Categorical features analysis
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        print(f"\nDistribution of {col}:\n")
        sns.countplot(data=df, x=col, palette='Set2')
        plt.title(f"Distribution of {col} in {dataset_name}")
        plt.xticks(rotation=45)
        plt.show()

    # Numerical features analysis
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_features:
        print(f"\nDistribution of {col}:\n")
        sns.histplot(data=df, x=col, kde=True, color='blue')
        plt.title(f"Distribution of {col} in {dataset_name}")
        plt.show()

    # Pairplot for numerical features
    print("\nPairplot for numerical features:\n")
    sns.pairplot(df.select_dtypes(include=['int64', 'float64']), diag_kind='kde')
    plt.title(f"Pairplot of Numerical Features in {dataset_name}")
    plt.show()

# Perform EDA on training and test datasets
perform_eda(train_df, "Training Dataset")
perform_eda(test_df, "Test Dataset")
