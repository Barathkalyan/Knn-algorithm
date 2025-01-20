# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Data preparation
X = np.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750]).reshape(-1, 1)
y = np.array([150, 200, 250, 275, 300, 325, 350, 375, 400, 425])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the KNN model
model = KNeighborsRegressor(n_neighbors=2)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Values', marker='o')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values', marker='x')
plt.title("KNN Regression: Actual vs Predicted Values")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.grid(True)
plt.show()
