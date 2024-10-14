import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_processing import preprocess_data

def visualize_clusters(file_path):
    # Load the model
    with open('./models/kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)

    # Preprocess the data
    X_scaled = preprocess_data(file_path)
    
    # Reduce dimensions for visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Predict clusters
    labels = kmeans.predict(X_scaled)
    
    # Plot the clusters
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('Customer Segments')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

if __name__ == "__main__":
    visualize_clusters('./data/customer_data.csv')
