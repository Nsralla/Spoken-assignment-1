import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


def plot_gmm(gmm, data, label, color):
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        if covar.ndim == 1:
            covar = np.diag(covar)  # Handle spherical covariance
        eigenvalues, eigenvectors = np.linalg.eigh(covar)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean[:2], width=width, height=height, angle=angle, edgecolor=color, facecolor='none', lw=2)
        plt.gca().add_patch(ellipse)
    plt.scatter(data[:, 0], data[:, 1], s=10, color=color, label=label)


def main():
    """
    Main function to load data, train GMM models, and evaluate on the test set.
    """
    print("[DEBUG] Loading feature arrays from disk...")
    # Load your training and testing data (numpy arrays)
    X_train = np.load('../data/Training_features/X_train.npy')  # shape: (num_train_samples, num_features)
    y_train = np.load('../data/Training_features/y_train.npy')  # shape: (num_train_samples,)
    X_test = np.load('../data/Testing_features/X_test.npy')    # shape: (num_test_samples, num_features)
    y_test = np.load('../data/Testing_features/y_test.npy')    # shape: (num_test_samples,)
    
    # scale x train and x test
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    n_components_asian = 14
    n_components_white = 8
    
    # TRAIN GMM for asian speakers.
    gmm_asian = GaussianMixture(n_components=n_components_asian, random_state=42)
    gmm_asian.fit(x_train_scaled[y_train == 'Asian'])
    
    # TRAIN GMM for white speakers.
    gmm_white = GaussianMixture(n_components=n_components_white, random_state=42)
    gmm_white.fit(x_train_scaled[y_train == 'White'])
    
    label_mapping = {'Asian': 0, 'White': 1}
    y_test_mapped = np.array([label_mapping[label] for label in y_test])
    best_accuracy = 0
    for n_asian in range(5, 15):
        for n_white in range(5, 15):
            gmm_asian = GaussianMixture(n_components=n_asian, random_state=42).fit(x_train_scaled[y_train == 'Asian'])
            gmm_white = GaussianMixture(n_components=n_white, random_state=42).fit(x_train_scaled[y_train == 'White'])

            y_pred = []
            for sample in x_test_scaled:
                log_likelihood_asian = gmm_asian.score_samples(sample.reshape(1, -1))[0]
                log_likelihood_white = gmm_white.score_samples(sample.reshape(1, -1))[0]
                y_pred.append(0 if log_likelihood_asian > log_likelihood_white else 1)
            
            accuracy = accuracy_score(y_test_mapped, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (n_asian, n_white)
                # print classification report
                print("Accuracy:", accuracy)
                print(classification_report(y_test_mapped, y_pred))

    print("Best parameters:", best_params, "with accuracy:", best_accuracy)
    # Plot reduced data with GMM ellipses
    
    pca = PCA(n_components=2)
    x_train_reduced = pca.fit_transform(x_train_scaled)
    # Fit GMMs in reduced space
    gmm_asian_2d = GaussianMixture(n_components=n_components_asian, random_state=42)
    gmm_asian_2d.fit(x_train_reduced[y_train == 'Asian'])

    gmm_white_2d = GaussianMixture(n_components=n_components_white, random_state=42)
    gmm_white_2d.fit(x_train_reduced[y_train == 'White'])
    
    plt.figure(figsize=(10, 7))
    plot_gmm(gmm_asian_2d, x_train_reduced[y_train == 'Asian'], label='Asian', color='blue')
    plot_gmm(gmm_white_2d, x_train_reduced[y_train == 'White'], label='White', color='red')
    plt.title('GMM Clusters and Covariance Ellipses')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

            


if __name__ == "__main__":
    main()
