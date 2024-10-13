# Import necessary libraries
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to plot results
def plot_results(y_test, y_pred, title):
    plt.scatter(y_test, y_pred, color='blue', marker='o', label='Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2, label='True values')
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.show()

# 1. Ridge Regression (L2 Regularization)
ridge_model = Ridge(alpha=1.0)  # alpha is the regularization strength
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Ridge Regression - Mean Squared Error: {mse_ridge}")

# Plot Ridge results
plot_results(y_test, y_pred_ridge, "Ridge Regression (L2)")

# 2. Lasso Regression (L1 Regularization)
lasso_model = Lasso(alpha=1.0)  # alpha is the regularization strength
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"Lasso Regression - Mean Squared Error: {mse_lasso}")

# Plot Lasso results
plot_results(y_test, y_pred_lasso, "Lasso Regression (L1)")

# 3. ElasticNet (Combination of L1 and L2 Regularization)
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio controls balance between L1 and L2
elastic_model.fit(X_train, y_train)
y_pred_elastic = elastic_model.predict(X_test)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
print(f"ElasticNet Regression - Mean Squared Error: {mse_elastic}")

# Plot ElasticNet results
plot_results(y_test, y_pred_elastic, "ElasticNet Regression (L1 & L2)")
