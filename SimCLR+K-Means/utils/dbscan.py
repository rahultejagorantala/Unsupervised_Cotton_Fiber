import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons  # Example dataset
import matplotlib.pyplot as plt
import os

def dbscan(input_dir):
    features = np.load(os.path.join(input_dir, 'features.npy')) # loading cropped data
    targets = np.load(os.path.join(input_dir, 'targets.npy')) # loading cropped data

    dbscan = DBSCAN(eps=0.2, min_samples=25)
    labels = dbscan.fit_predict(features)
    print(labels, targets)

def example():
# Generate example data (replace this with your own data)
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

    # Initialize DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)

    # Fit the model and predict clusters
    labels = dbscan.fit_predict(X)
    print(labels)
    # Visualize the clusters
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    for label in unique_labels:
        if label == -1:
            color = 'gray'  # Noise points
        else:
            color = colors(label)
        
        plt.scatter(X[labels == label, 0], X[labels == label, 1], color=color, label=f'Cluster {label}')

    plt.legend()
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

root = r"C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master"
basefolder = r"results\cotton\pretext\embeddings"
input_dir = os.path.join(root, basefolder)

input_dir = r"C:\Users\AVLguest\Desktop\cotton\CROPPED TO FIBER AND IMAGE POSITIVE PAIRS\image_positive_pairs\pretext-COTTON DATASET, 64 BATCH SIZE, SAME SAMPLE IMAGE POSITIVE PAIRS (SINGLE IMAGE FROM VIDEO)\embeddings"
# dbscan(input_dir)
example()
