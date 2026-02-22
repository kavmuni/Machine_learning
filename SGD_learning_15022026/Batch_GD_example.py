import numpy as np
import pandas as pd

# Sample data: y ≈ 4 + 3x + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term (intercept)
X_b = np.c_[np.ones((100, 1)), X]
print(type(X_b), type(y))
print('First 5 rows of X_b:\n', X_b[:5])
print('First 5 values of y:\n', y[:5])
X_b_df = pd.DataFrame(X_b, columns=['Intercept', 'X'])
print(X_b_df.head(5))


# BGD function
def batch_gd(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(y)
    theta = np.random.randn(2, 1)  # Random init: intercept, slope

    for iteration in range(n_iterations):
        # Compute gradients over full batch
        gradients = 2 / m * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradients  # Update

        # Print MSE every 100 iterations
        if iteration % 100 == 0:
            y_pred = X.dot(theta)
            mse = np.mean((y_pred - y) ** 2)
            print(f'Iteration {iteration}, MSE: {mse:.4f}')

    return theta


# Run it
theta_final = batch_gd(X_b, y)
print('Final parameters (intercept, slope):', theta_final.flatten())
