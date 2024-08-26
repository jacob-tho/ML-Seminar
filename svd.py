import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize


'''
Dieser Code wurde nicht selber geschrieben. Inspiration aus Numerik-Übungsaufgabe
'''

def svd_approximation(image, rank):
    """
    Perform SVD and return an approximation of the image with the given rank for each color channel.
    """
    approx_image = np.zeros_like(image)
    for i in range(3):  # Apply SVD to each color channel
        U, S, Vt = np.linalg.svd(image[:, :, i], full_matrices=False)
        S = np.diag(S)
        approx_image[:, :, i] = np.dot(U[:, :rank], np.dot(S[:rank, :rank], Vt[:rank, :]))
    return np.clip(approx_image, 0, 1)  # Clip values to be in the valid range [0, 1]

def plot_image(image, title, subplot):
    ax = plt.subplot(*subplot)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

def process_image(filename):
    """
    Load an image, perform SVD approximations, and display the results.
    """
    # Load the image
    image = io.imread(filename) / 255.0  # Normalize pixel values to [0, 1]

    # Resize the image if it's too large (optional)
    max_dimension = 512
    if image.shape[0] > max_dimension or image.shape[1] > max_dimension:
        image = resize(image, (max_dimension, max_dimension), anti_aliasing=True)

    plt.figure(figsize=(10, 7))
    plot_image(image, 'Original', (2, 3, 1))

    # SVD-Approximation bei unterschiedlichen Rängen
    ranks = [1, 2, 3, 10, 40, 150]
    for i, rank in enumerate(ranks):
        approx_image = svd_approximation(image, rank)
        plot_image(approx_image, f'Rang {rank}', (2, 3, i+1))

    plt.tight_layout()
    plt.show()

# Usage example
filename = r"C:\Users\Thorwarth\Downloads\images\images\kandinsky.jpg"  # Replace with your image path
process_image(filename)
