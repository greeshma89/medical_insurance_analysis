import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def perform_eda(df):
    """Generates visual insights from the data."""
    print("--- Performing EDA ---")
    sns.set_theme(style="whitegrid")
    
    # 1. Distribution of Charges (The Target)
    plt.figure(figsize=(10, 5))
    sns.histplot(df['charges'], kde=True, color='teal')
    plt.title('Distribution of Medical Charges')
    plt.savefig('eda_distribution.png')
    plt.close()

    # 2. Impact of Smoking on Charges
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='smoker', y='charges', data=df, palette='Set2')
    plt.title('Medical Charges: Smokers vs Non-Smokers')
    plt.xticks([0, 1], ['Non-Smoker', 'Smoker'])
    plt.savefig('eda_smoking_impact.png')
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig('eda_heatmap.png')
    plt.close()
    print("EDA Visuals saved as PNG files.")

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.strip()
    
    # Mapping
    df['smoker'] = df['smoker'].str.strip().map({'yes': 1, 'no': 0})
    df['sex'] = df['sex'].str.strip().map({'female': 1, 'male': 2})
    df['region'] = df['region'].str.strip().map({'northwest': 1, 'northeast': 2,'southwest': 3, 'southeast': 4})
    
    return df

def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "R2 Score": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
    }

def main():
    # TIP: For GitHub, keep the CSV in the same folder and use just the filename
    DATA_PATH = "C:/Users/grees/Downloads/Project/Medical_insurance.csv" 
    df = load_and_clean_data(DATA_PATH)
    
    # Run EDA
    perform_eda(df)
    
    y = df['charges']
    results = []

    # Model Training Logic
    # 1. Single Variable (Smoker)
    X_s = df[['smoker']]
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)
    m_s = LinearRegression().fit(X_train, y_train)
    results.append(evaluate_model("Linear (Single-Smoker)", y_test, m_s.predict(X_test)))

    # 2. Multi-Variable (Smoker, Age, BMI)
    X_m = df[['smoker','age', 'bmi']]
    X_train, X_test, y_train, y_test = train_test_split(X_m, y, test_size=0.2, random_state=42)
    m_m = LinearRegression().fit(X_train, y_train)
    results.append(evaluate_model("Linear (Multi-Variable)", y_test, m_m.predict(X_test)))

    # 3. Ridge Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    m_r = Ridge(alpha=1.0).fit(X_train_sc, y_train)
    results.append(evaluate_model("Ridge Regression", y_test, m_r.predict(X_test_sc)))

    summary_df = pd.DataFrame(results)
    print("\n--- Model Performance Summary ---")
    print(summary_df)

if __name__ == "__main__":
    main()