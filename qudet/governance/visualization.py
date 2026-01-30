
import matplotlib.pyplot as plt
import numpy as np

def plot_reduction_2d(original_data: np.ndarray, reduced_data: np.ndarray):
    """
    Visualizes the 'Smart Funnel' effect.
    Plots the massive original dataset (grey) vs the Quantum Coreset (red).
    Assumes data is already 2D or uses first 2 columns.
    """
    plt.figure(figsize=(10, 6))
    
    limit = min(len(original_data), 5000)
    plt.scatter(original_data[:limit, 0], original_data[:limit, 1], 
                c='lightgrey', alpha=0.5, label='Classical Big Data')
    
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                c='red', s=100, marker='x', label='Quantum Coreset')
    
    plt.title("Visualizing Data Reduction for QPU Handoff")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_kernel_matrix(matrix: np.ndarray):
    """
    Plots the Quantum Kernel Matrix (Gram Matrix).
    Diagonal should be 1.0 (self-similarity).
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Quantum Fidelity')
    plt.title("Quantum Kernel Matrix")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.show()
