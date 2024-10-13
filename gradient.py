import numpy as np

# Generate some sample data (you can replace this with any digits or dataset)
X = np.array([1, 2, 3, 4, 5])  # Input data (features)
y = np.array([5, 7, 9, 11, 13])  # Output data (labels)

# Initialize parameters (weights and bias)
w = 0.0  # weight (slope)
b = 0.0  # bias (intercept)

# Hyperparameters
learning_rate = 0.01  # Learning rate (alpha)
iterations = 1000  # Number of iterations

# Function to compute Mean Squared Error
def compute_cost(X, y, w, b):
    m = len(y)
    total_cost = np.sum((w * X + b - y) ** 2) / (2 * m)
    return total_cost

# Function to compute gradients and update parameters
def gradient_descent(X, y, w, b, learning_rate, iterations):
    m = len(y)
    cost_history = []  # To keep track of cost function value

    for i in range(iterations):
        # Calculate predictions
        y_pred = w * X + b

        # Calculate gradients
        dw = (1/m) * np.sum((y_pred - y) * X)  # Gradient with respect to w
        db = (1/m) * np.sum(y_pred - y)        # Gradient with respect to b

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Compute cost for this iteration
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        # Print cost every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}, w {w}, b {b}")

    return w, b, cost_history

# Run Gradient Descent
w_final, b_final, cost_history = gradient_descent(X, y, w, b, learning_rate, iterations)

print(f"Final values after Gradient Descent: w = {w_final}, b = {b_final}")

# Output predicted values
y_pred_final = w_final * X + b_final
print(f"Predicted values: {y_pred_final}")
print(f"Actual values: {y}")
