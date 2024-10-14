import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Select features for clustering
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

if __name__ == "__main__":
    X_scaled = preprocess_data('./data/customer_data.csv')
    print("Data preprocessing complete.")