import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
C = np.cov(x, rowvar=False)
print(C) #Zum Erkennen der Eigentschaft: Kovarianzmatrix ist symmetrisch
T, s, Tinv = np.linalg.svd(C) #Da Symmetrisch ist SVD auch Eigenzerlegung (Tinv = T.T)
pca_x = np.matmul(Tinv,x.T).T #Berechnung des neuen Feature-Space

fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
ax.view_init(15, 75)
ax.scatter3D(pca_x[:,0], pca_x[:,1], pca_x[:,2], c=iris.target)
ax.set_title("Die ersten drei Hauptkomponenten des Iris-Datensatzes")
plt.show()


fig, ax = plt.subplots()
ax.scatter(pca_x[:,0], pca_x[:,1], c=iris.target, alpha=0.6)
ax.set_title("Die ersten zwei Hauptkomponenten des Iris-Datensatzes")
plt.show()
