import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Erstellen von Beispiel-Daten
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 1 * (0.5 - np.random.rand(16))  # Hinzufügen von Rauschen

# Plotten der Originaldaten
plt.figure(figsize=(14, 7))
plt.scatter(X, y, color='black', label='Datenpunkte')

# Funktion zur Erstellung von Modellen und Plotten der Ergebnisse
def plot_regression(degree, label, color):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, color=color, label=label)

plot_regression(1, 'Underfitting (Grad 1)', 'blue')

plot_regression(3, 'Optimale Lösung (Grad 3)', 'red')

plot_regression(10, 'Overfitting (Grad 10)', 'green')

plot_regression(23, 'Extreme Überanpassung', 'purple')

# Plot-Einstellungen
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression: Underfitting, Overfitting, Optimale Lösung und Extreme Überanpassung')
plt.legend()
plt.show()
