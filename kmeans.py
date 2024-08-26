import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def initialize_centroids(X, k):
    random_indices = np.random.permutation(X.shape[0])
    centroids = X[random_indices[:k]]
    return centroids


def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    return clusters

# Funktion, um Centroids nach Mittelwertberechnung zu verschieben
def update_centroids(X, clusters, k):
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids

# within-cluster sum of squares (WCSS)
def calculate_wcss(X, clusters, centroids):
    wcss = np.sum((X - centroids[clusters])**2)
    return wcss

# K-means clustering Algorithmus
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # Subplots
    plot_idx = 0

    for i in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        #Plotte jede zweite Iteration (für Übersichtlichkeit)
        if i % 2 == 0 and plot_idx < 6:
            ax = axes[plot_idx // 2, plot_idx % 2]
            plot_clusters(X, clusters, centroids, i, ax)
            plot_idx += 1

    plt.tight_layout()
    plt.show()
    return clusters, centroids


def plot_clusters(X, clusters, centroids, iteration, ax):
    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
    ax.set_title(f'Iteration {iteration}')

# Bei wenig Änderung (="Knick") Wert wählen
def plot_elbow(X, max_k=10):
    wcss = []
    for k in range(1, max_k+1):
        clusters, centroids = kmeans(X, k, max_iters=10)  # Reduce iterations for elbow plot speed
        wcss.append(calculate_wcss(X, clusters, centroids))
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k+1), wcss, 'bo-', markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method For Optimal k')
    plt.show()

# Sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plot_elbow(X, max_k=6)
k = 4  # Ergebnis aus Ellbow-Graph


clusters, centroids = kmeans(X, k)
