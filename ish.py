from sklearn.datasets import load_iris
# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Splitting the data into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Fitting the Logistic Regression model
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(penalty='l2', tol=1e-3, verbose = 1, multi_class='multinomial', solver='lbfgs', max_iter=100, C=10)
log.fit(X_train, y_train)
# Predict the probabilities for test set
y_proba = log.predict_proba(X_test)
print(y_proba.shape)
print(y_proba[:6])
# Prediction for test set
y_pred = log.predict(X_test)
print(y_pred.shape)
print(y_pred[:6])
# Evaluate model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Calculating the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Classification report
print(classification_report(y_test, y_pred))
# Confusion matrix
print(confusion_matrix(y_test, y_pred)) 
# Checking for cross-validation score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(log, X, y, cv=5)
print(f"Cross-validation score: {scores}")
print(f"Average accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
# Visualizing the decision boundaries using PCA (Principal Component Analysis)
# PCA is used here to reduce the dimensionality of the data from 4 features to 2 principal components
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# Initialize PCA to reduce data to 2 dimensions
pca = PCA(n_components=2)
# Plot the data points in the 2D PCA space, colored by their target class
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('Iris Dataset - First Two Principal Components')
plt.show()

import pandas as pd
# Feature importance
coef = pd.DataFrame(
    log.coef_,
    columns=iris.feature_names,
    index=['class 0 vs rest', 'class 1 vs rest', 'class 2 vs rest']
)
plt.figure(figsize=(10,6))
sns.heatmap(coef, annot=True, cmap='RdBu')
plt.title('Feature Importance for Each Class')
plt.show()