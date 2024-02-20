import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

filename = "ApneaData.pkl"

# Load data from the pickle file
with open(filename, 'rb') as f:
    data = pickle.load(f)

# Split data into features and classes
features = np.array([row[:-1] for row in data])
classes = np.array([row[-1] for row in data])

# Dimensionality reduction with TSNE
reduced_features = TSNE(n_components=3).fit_transform(features)

# Create scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot data points with different colors for each class
for label, color in zip((0, 1), ('g', 'r')):
    indices = classes == label
    ax.scatter(*reduced_features[indices].T, c=color, label=label)

# Show legend and plot
ax.legend()
plt.show()