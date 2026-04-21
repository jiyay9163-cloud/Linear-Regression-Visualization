import numpy as np
import matplotlib.pyplot as plt

# Take user input
x = list(map(float, input("Enter X values (space separated): ").split()))
y = list(map(float, input("Enter Y values (space separated): ").split()))

x = np.array(x)
y = np.array(y)

# Sort data (for clean graph)
sorted_indices = np.argsort(x)
x = x[sorted_indices]
y = y[sorted_indices]

# Calculate slope (m) and intercept (b)
m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
b = np.mean(y) - m * np.mean(x)

# Predictions for given data
y_pred = m * x + b

# Error
mse = np.mean((y - y_pred) ** 2)

# Predict for new value
x_new = float(input("Enter a new X value to predict Y: "))
y_new = m * x_new + b

# Plot
plt.scatter(x, y, label="Data Points")
plt.plot(x, y_pred, label="Best Fit Line")

# Highlight predicted point
plt.scatter(x_new, y_new, marker='x', s=100, label="Predicted Point")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression Visualizer: How Machines Learn from Data")
plt.legend()

plt.show()

print("Slope (m):", m)
print("Intercept (b):", b)
print("Mean Squared Error:", mse)
print(f"Predicted value for X = {x_new} is Y = {y_new}")
