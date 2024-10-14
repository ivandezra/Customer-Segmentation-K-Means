import pickle
from sklearn.cluster import KMeans
from data_processing import preprocess_data

def perform_clustering(file_path, n_clusters=5):
    # Preprocess the data
    X_scaled = preprocess_data(file_path)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    # Save the model
    with open('./models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    print(f"K-Means clustering complete. Model saved with {n_clusters} clusters.")

if __name__ == "__main__":
    perform_clustering('./data/customer_data.csv')