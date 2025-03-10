import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from my_lr import linear_regression, A, B

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature column
X_flat = X.flatten()  # For your implementation
y = 2 * X.flatten() + 1 + np.random.randn(100) * 2  # Target: y = 2x + 1 + noise

# Scikit-learn implementation
model = LinearRegression()
model.fit(X, y)
sklearn_coef = model.coef_[0]
sklearn_intercept = model.intercept_
sklearn_predictions = model.predict(X)

# Your implementation
your_slope = A(X_flat, y)
your_intercept = B(X_flat, y)
your_predictions = [linear_regression(x_i, X_flat, y) for x_i in X_flat]

# Print results for comparison
print("Scikit-learn implementation:")
print(f"Slope: {sklearn_coef:.6f}")
print(f"Intercept: {sklearn_intercept:.6f}")
print(f"MSE: {mean_squared_error(y, sklearn_predictions):.6f}")
print(f"R²: {r2_score(y, sklearn_predictions):.6f}")
print()

print("Your implementation:")
print(f"Slope: {your_slope:.6f}")
print(f"Intercept: {your_intercept:.6f}")
print(f"MSE: {mean_squared_error(y, your_predictions):.6f}")
print(f"R²: {r2_score(y, your_predictions):.6f}")

# Visualize both implementations
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
plt.plot(X, sklearn_predictions, color='red', linewidth=2, label='Scikit-learn')
plt.plot(X, your_predictions, color='green', linewidth=2, linestyle='--', label='Your implementation')
plt.title('Linear Regression Comparison')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check a specific prediction
test_x = 5.5
sklearn_pred = model.predict([[test_x]])[0]
your_pred = linear_regression(test_x, X_flat, y)
print(f"\nPrediction for x = {test_x}:")
print(f"Scikit-learn: {sklearn_pred:.6f}")
print(f"Your implementation: {your_pred:.6f}")
print(f"Difference: {abs(sklearn_pred - your_pred):.6f}")