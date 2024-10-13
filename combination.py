import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data generation: replace with your dataset
X = np.array([[1], [2], [3], [4], [5]])  # Input features
y = np.array([5, 7, 9, 11, 13])  # Output labels

# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Base class for Gradient Descent
class GradientDescentModel:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0
        self.b = 0

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def compute_cost(self, X, y):
        m = len(y)
        y_pred = self.predict(X)
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        return cost

    def update_weights(self, X, y):
        pass

    def fit(self, X, y):
        m = len(y)
        cost_history = []

        for i in range(self.iterations):
            # Calculate gradients and update weights
            self.update_weights(X, y)
            # Calculate and store the cost
            cost = self.compute_cost(X, y)
            cost_history.append(cost)

            # Print cost every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost}")

        return cost_history

# Ridge Regression (L2 Regularization)
class RidgeRegression(GradientDescentModel):
    def __init__(self, learning_rate=0.01, iterations=1000, alpha=1.0):
        super().__init__(learning_rate, iterations)
        self.alpha = alpha

    def update_weights(self, X, y):
        m = len(y)
        y_pred = self.predict(X)
        dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (self.alpha / m) * self.w
        db = (1 / m) * np.sum(y_pred - y)

        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

# Lasso Regression (L1 Regularization)
class LassoRegression(GradientDescentModel):
    def __init__(self, learning_rate=0.01, iterations=1000, alpha=1.0):
        super().__init__(learning_rate, iterations)
        self.alpha = alpha

    def update_weights(self, X, y):
        m = len(y)
        y_pred = self.predict(X)
        dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (self.alpha / m) * np.sign(self.w)
        db = (1 / m) * np.sum(y_pred - y)

        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

# ElasticNet (Combination of L1 and L2)
class ElasticNetRegression(GradientDescentModel):
    def __init__(self, learning_rate=0.01, iterations=1000, alpha=1.0, l1_ratio=0.5):
        super().__init__(learning_rate, iterations)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def update_weights(self, X, y):
        m = len(y)
        y_pred = self.predict(X)
        dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (self.alpha / m) * (
            self.l1_ratio * np.sign(self.w) + (1 - self.l1_ratio) * self.w)
        db = (1 / m) * np.sum(y_pred - y)

        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

# Function to train and evaluate all models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Ridge": RidgeRegression(learning_rate=0.01, iterations=1000, alpha=0.1),
        "Lasso": LassoRegression(learning_rate=0.01, iterations=1000, alpha=0.1),
        "ElasticNet": ElasticNetRegression(learning_rate=0.01, iterations=1000, alpha=0.1, l1_ratio=0.5)
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        cost_history = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        print(f"{name} Mean Squared Error: {mse}")

        # Plot cost history
        plt.plot(cost_history, label=name)

    plt.title("Cost History")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

# Evaluate models on the dataset
evaluate_models(X_train, X_test, y_train, y_test)
