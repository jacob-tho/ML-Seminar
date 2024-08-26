import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# Olivetti-Dataset
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# Erstes Bild
first_image = faces.images[0]

plt.figure(figsize=(6, 6))
plt.imshow(first_image, cmap='gray')
plt.title('Erstes Bild  Olivetti Faces Dataset')
plt.xticks(())
plt.yticks(())
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Genauigkeit und Zeit in Abhängigkeit der PCA
def compute_pca_accuracies_and_time(X_train, X_test, y_train, y_test, max_n_components):
    svm_accuracies = []
    knn_accuracies = []
    times = []
    components_range = list(range(1, max_n_components))
    for n_components in components_range:
        start_time = time.time()

        pca = PCA(n_components=n_components, whiten=True).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # SVM
        svm_pca = SVC(kernel='linear', random_state=42)
        svm_pca.fit(X_train_pca, y_train)
        y_pred_svm_pca = svm_pca.predict(X_test_pca)
        accuracy_svm_pca = accuracy_score(y_test, y_pred_svm_pca)

        # KNN als Base-line Algorithmus
        knn_pca = KNeighborsClassifier(n_neighbors=5)
        knn_pca.fit(X_train_pca, y_train)
        y_pred_knn_pca = knn_pca.predict(X_test_pca)
        accuracy_knn_pca = accuracy_score(y_test, y_pred_knn_pca)

        end_time = time.time()

        svm_accuracies.append(accuracy_svm_pca)
        knn_accuracies.append(accuracy_knn_pca)
        times.append(end_time - start_time)

    return components_range, svm_accuracies, knn_accuracies, times

# Genauigkeit und Zeit berechnen
max_n_components = min(X_train.shape[0], X_train.shape[1])
components_range, svm_accuracies, knn_accuracies, times = compute_pca_accuracies_and_time(X_train, X_test, y_train, y_test, max_n_components)


plt.figure(figsize=(18, 6))
# Plot für SVM vs KNN
plt.subplot(1, 3, 1)
plt.plot(components_range, svm_accuracies, marker='o', label='SVM')
plt.plot(components_range, knn_accuracies, marker='x', label='KNN')
plt.title('Genauigkeit vs. Anzahl Principal Components')
plt.xlabel('Anzahl Principal Components')
plt.ylabel('Genauigkeit')
plt.legend()
plt.grid(True)

# Plot für Zeit
plt.subplot(1, 3, 2)
plt.plot(components_range, times, marker='o', color='r')
plt.title('Computation Time vs. Anzahl Principal Components')
plt.xlabel('Anzahl Principal Components')
plt.ylabel('Computation Time (Sekunden)')
plt.grid(True)

plt.tight_layout()
plt.show()
