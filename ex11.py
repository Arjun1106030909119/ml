# SVM on Iris (visualize decision boundary using first two features)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]   # use only the first two features for visualization
y = iris.target

print("Iris data loaded successfully.")
print("Data shape (X):", X.shape)
print("Target shape (y):", y.shape)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training data shape (X_train):", X_train.shape)
print("Test data shape (X_test):", X_test.shape)

# Initialize the Support Vector Classifier with a linear kernel
model = SVC(kernel='linear')
print("SVC model initialized with a linear kernel.")

# Train the model with the training data
model.fit(X_train, y_train)
print("Model trained on the training data.")

# Make predictions on the test set
y_pred = model.predict(X_test)
print("Predictions on test set:", y_pred)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")

# Visualizing the decision boundary for the SVC model
h = 0.02  # step size in the mesh

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Get predictions for each point in the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Linear Kernel Decision Boundary')
plt.show()