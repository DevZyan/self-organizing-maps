"""
Self-Organizing Maps (SOM) Implementation
Author: Nayan Pal (24AI10039)
Date: November 2025

This script contains a from-scratch implementation of Self-Organizing Maps
using NumPy, along with experiments on Iris and synthetic blob datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import MinMaxScaler

# Set random seed for reproducibility
np.random.seed(42)


class SOM_FromScratch:
    """Self-Organizing Map implementation from scratch using NumPy"""

    def __init__(self, grid_size, input_dim, learning_rate=0.5, sigma=None, n_iterations=5000):
        """
        Initialize SOM

        Parameters:
        - grid_size: tuple (height, width) of the SOM grid
        - input_dim: dimensionality of input data
        - learning_rate: initial learning rate
        - sigma: initial neighborhood radius (defaults to max grid dimension / 2)
        - n_iterations: number of training iterations
        """
        self.grid_height, self.grid_width = grid_size
        self.input_dim = input_dim
        self.n_iterations = n_iterations
        self.alpha_0 = learning_rate

        # Set initial sigma if not provided
        if sigma is None:
            self.sigma_0 = max(self.grid_height, self.grid_width) / 2.0
        else:
            self.sigma_0 = sigma

        # Initialize weights randomly in [0, 1]
        self.weights = np.random.rand(self.grid_height, self.grid_width, input_dim)

        # Create coordinate grid for neurons
        self.neuron_positions = np.array([[i, j] for i in range(self.grid_height) 
                                          for j in range(self.grid_width)])

    def _find_bmu(self, x):
        """Find the Best Matching Unit for input x"""
        # Compute Euclidean distances to all neurons
        distances = np.linalg.norm(self.weights - x, axis=2)
        # Find index of minimum distance
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def _calculate_neighborhood(self, bmu_idx, iteration):
        """Calculate neighborhood function for all neurons"""
        # Current learning rate and sigma (exponential decay)
        alpha_t = self.alpha_0 * np.exp(-iteration / self.n_iterations)
        sigma_t = self.sigma_0 * np.exp(-iteration / self.n_iterations)

        # BMU position
        bmu_pos = np.array([bmu_idx[0], bmu_idx[1]])

        # Calculate distances from BMU to all neurons on the grid
        distances_squared = np.sum((self.neuron_positions - bmu_pos) ** 2, axis=1)

        # Gaussian neighborhood function
        neighborhood = np.exp(-distances_squared / (2 * sigma_t ** 2))
        neighborhood = neighborhood.reshape(self.grid_height, self.grid_width)

        return alpha_t, neighborhood

    def train(self, X, verbose=True):
        """Train the SOM on dataset X"""
        n_samples = X.shape[0]

        for iteration in range(self.n_iterations):
            # Random sample
            idx = np.random.randint(0, n_samples)
            x = X[idx]

            # Find BMU
            bmu_idx = self._find_bmu(x)

            # Calculate neighborhood and learning rate
            alpha_t, neighborhood = self._calculate_neighborhood(bmu_idx, iteration)

            # Update weights
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    self.weights[i, j] += alpha_t * neighborhood[i, j] * (x - self.weights[i, j])

            if verbose and (iteration + 1) % 1000 == 0:
                print(f"  Iteration {iteration + 1}/{self.n_iterations}")

    def predict(self, X):
        """Map input samples to their BMUs"""
        bmus = []
        for x in X:
            bmu_idx = self._find_bmu(x)
            bmus.append(bmu_idx)
        return np.array(bmus)

    def quantization_error(self, X):
        """Calculate average quantization error"""
        errors = []
        for x in X:
            bmu_idx = self._find_bmu(x)
            error = np.linalg.norm(x - self.weights[bmu_idx])
            errors.append(error)
        return np.mean(errors)

    def get_u_matrix(self):
        """Calculate U-matrix (unified distance matrix)"""
        u_matrix = np.zeros((self.grid_height, self.grid_width))

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                neighbors = []
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_height and 0 <= nj < self.grid_width:
                            dist = np.linalg.norm(self.weights[i, j] - self.weights[ni, nj])
                            neighbors.append(dist)
                u_matrix[i, j] = np.mean(neighbors) if neighbors else 0

        return u_matrix

    def get_hit_map(self, X):
        """Calculate hit map (number of samples mapped to each neuron)"""
        hit_map = np.zeros((self.grid_height, self.grid_width))
        bmus = self.predict(X)
        for bmu in bmus:
            hit_map[bmu[0], bmu[1]] += 1
        return hit_map


def load_and_preprocess_data():
    """Load and normalize datasets"""
    # Load Iris dataset
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    # Create synthetic blob dataset
    X_blobs, y_blobs = make_blobs(n_samples=300, n_features=2, centers=4,
                                   cluster_std=0.5, random_state=42)

    # Normalize to [0, 1]
    scaler_iris = MinMaxScaler()
    X_iris_norm = scaler_iris.fit_transform(X_iris)

    scaler_blobs = MinMaxScaler()
    X_blobs_norm = scaler_blobs.fit_transform(X_blobs)

    return X_iris_norm, y_iris, X_blobs_norm, y_blobs


def calculate_cluster_purity(som, X, y):
    """Calculate cluster purity for labeled data"""
    bmus = som.predict(X)
    neuron_labels = {}

    # Collect labels for each neuron
    for i, bmu in enumerate(bmus):
        key = tuple(bmu)
        if key not in neuron_labels:
            neuron_labels[key] = []
        neuron_labels[key].append(y[i])

    # Count correct assignments
    correct = 0
    for i, bmu in enumerate(bmus):
        key = tuple(bmu)
        labels = neuron_labels[key]
        majority_label = max(set(labels), key=labels.count)
        if majority_label == y[i]:
            correct += 1

    return correct / len(y)


def run_experiments():
    """Run all SOM experiments"""
    print("Loading and preprocessing data...")
    X_iris, y_iris, X_blobs, y_blobs = load_and_preprocess_data()

    print("\nExperiment 1: Grid size comparison on Iris")
    grid_sizes = [(6, 6), (8, 8), (10, 10)]
    results = []

    for grid_size in grid_sizes:
        print(f"  Training {grid_size[0]}×{grid_size[1]} SOM...")
        som = SOM_FromScratch(grid_size, X_iris.shape[1], learning_rate=0.5,
                             sigma=max(grid_size)/2, n_iterations=5000)
        som.train(X_iris, verbose=False)
        qe = som.quantization_error(X_iris)
        results.append((grid_size, qe))
        print(f"  QE: {qe:.4f}")

    print("\nExperiment 2: Iris dataset (8×8 SOM)")
    som_iris = SOM_FromScratch((8, 8), X_iris.shape[1], learning_rate=0.5,
                               sigma=4.0, n_iterations=5000)
    som_iris.train(X_iris, verbose=False)
    qe_iris = som_iris.quantization_error(X_iris)
    purity_iris = calculate_cluster_purity(som_iris, X_iris, y_iris)
    print(f"  QE: {qe_iris:.4f}, Purity: {purity_iris:.2%}")

    print("\nExperiment 3: Blob dataset (10×10 SOM)")
    som_blobs = SOM_FromScratch((10, 10), X_blobs.shape[1], learning_rate=0.5,
                                sigma=5.0, n_iterations=5000)
    som_blobs.train(X_blobs, verbose=False)
    qe_blobs = som_blobs.quantization_error(X_blobs)
    purity_blobs = calculate_cluster_purity(som_blobs, X_blobs, y_blobs)
    print(f"  QE: {qe_blobs:.4f}, Purity: {purity_blobs:.2%}")

    return som_iris, som_blobs, X_iris, y_iris, X_blobs, y_blobs


if __name__ == "__main__":
    # Run all experiments
    som_iris, som_blobs, X_iris, y_iris, X_blobs, y_blobs = run_experiments()

    print("\n✓ All experiments completed!")
    print("\nTo generate visualizations, run the visualization script.")
