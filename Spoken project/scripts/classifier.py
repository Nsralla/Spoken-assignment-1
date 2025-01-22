from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  classification_report, accuracy_score
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# mapping for labels
label_mapping = {'Asian': 0, 'White': 1}

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='joblib.externals.loky.backend.context')

# Load feature arrays
X_train = np.load('../data/Training_features/X_train.npy')
y_train = np.load('../data/Training_features/y_train.npy')

# (Repeat for validation/test sets)
X_test = np.load(r'../data/Testing_features/X_test.npy')
y_test = np.load(r'../data/Testing_features/y_test.npy')
y_test_mapped = np.array([label_mapping[label] for label in y_test])

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

# seperate data by class
x_train_asian = x_train_scaled[y_train == 'Asian']
x_train_white = x_train_scaled[y_train == 'White']

# number of clusters for each class
n_clusters_asian = 9
n_clusters_white = 8

# apply k means for each class
kmeans_asian = KMeans(n_clusters=n_clusters_asian, random_state=42)
kmeans_white = KMeans(n_clusters=n_clusters_white, random_state=42)

kmeans_asian.fit(x_train_asian)
kmeans_white.fit(x_train_white)

centroids_asian = kmeans_asian.cluster_centers_
centroids_white = kmeans_white.cluster_centers_


# combine centroids from both classes
all_centroids = np.vstack((centroids_asian, centroids_white))

# labels for centoids (o asian, 1 white)
centroid_labels = np.array([0]*n_clusters_asian + [1]*n_clusters_white)

# predict for each test sample
y_pred = []
for sample in x_test_scaled:
    distances = np.linalg.norm(all_centroids - sample, axis=1)
    closest_centroid = np.argmin(distances)
    y_pred.append(centroid_labels[closest_centroid])

# evaluate the model
print("accuracy: ", accuracy_score(y_test_mapped, y_pred))
print("\n classification report: \n", classification_report(y_test_mapped, y_pred))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensionality to 2D for visualization
pca = PCA(n_components=2)
x_train_reduced = pca.fit_transform(x_train_scaled)

# Transform centroids to 2D
centroids_asian_reduced = pca.transform(centroids_asian)
centroids_white_reduced = pca.transform(centroids_white)

# Plot training data
plt.figure(figsize=(10, 7))
plt.scatter(
    x_train_reduced[y_train == 'Asian', 0],
    x_train_reduced[y_train == 'Asian', 1],
    label='Asian',
    alpha=0.5
)
plt.scatter(
    x_train_reduced[y_train == 'White', 0],
    x_train_reduced[y_train == 'White', 1],
    label='White',
    alpha=0.5
)

# Plot centroids
plt.scatter(
    centroids_asian_reduced[:, 0],
    centroids_asian_reduced[:, 1],
    color='red',
    marker='x',
    s=100,
    label='Asian Centroids'
)
plt.scatter(
    centroids_white_reduced[:, 0],
    centroids_white_reduced[:, 1],
    color='blue',
    marker='x',
    s=100,
    label='White Centroids'
)

# Customize plot
plt.title("Clusters and Centroids in 2D Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()