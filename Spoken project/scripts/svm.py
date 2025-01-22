import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load feature arrays
X_train = np.load('../data/Training_features/X_train.npy')
y_train = np.load('../data/Training_features/y_train.npy')
X_test = np.load(r'../data/Testing_features/X_test.npy')
y_test = np.load(r'../data/Testing_features/y_test.npy')

# (Repeat for validation/test sets)
X_test = np.load(r'../data/Testing_features/X_test.npy')
y_test = np.load(r'../data/Testing_features/y_test.npy')

y_mapping = {'Asian': 0, 'White': 1}
y_train = np.array([y_mapping[label] for label in y_train])
y_test = np.array([y_mapping[label] for label in y_test])

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# svm = SVC(kernel='rbf', gamma='auto')
# svm.fit(X_train, y_train)

# y_pred_val = svm.predict(X_test)
# acc_val_svm = accuracy_score(y_test, y_pred_val)
# print(f"SVM Validation Accuracy = {acc_val_svm:.2f}")
# print("\nClassification Report:\n", classification_report(y_test, y_pred_val))

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# PERFORM GRID SEARCH
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=3,verbose=2)
grid.fit(X_train,y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best estimator: {grid.best_estimator_}")

# predict with best model
best_svm_model = grid.best_estimator_
y_pred = best_svm_model.predict(X_test)

# print classification report
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def plot_decision_boundary(model, X, y):
    # Create a mesh to plot the decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

if __name__ == "__main__":
    # Reduce dimensionality to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    
    # Train the best model on reduced data
    best_svm_model.fit(X_train_reduced, y_train)
    
    # Plot decision boundary
    plot_decision_boundary(best_svm_model, X_test_reduced, y_test)


