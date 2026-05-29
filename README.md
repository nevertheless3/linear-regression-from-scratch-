# Linear Regression from Scratch

A clean and educational implementation of **Linear Regression** using Gradient Descent with early stopping and live training visualization. Built with pure NumPy and Matplotlib.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![NumPy](https://img.shields.io/badge/Numpy-1.2+-brightgreen)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0+-orange)

## Overview

This project implements **Linear Regression** from scratch with the following features:
- Mean Squared Error (MSE) loss
- R² Score evaluation
- Gradient Descent optimization
- Early stopping with patience
- Real-time training process visualization

---

## Mathematical Background

### Model
$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b
$$

### Cost Function (Mean Squared Error)
$$
J(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

### Gradients
$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})
$$
$$
\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
$$

### R² Score (Coefficient of Determination)
$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

---

## Hyperparameters

<h2>Hyperparameters</h2>

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <thead>
    <tr style="background-color: #2a2a2a; color: white;">
      <th style="padding: 12px; text-align: left; border: 1px solid #444;">Parameter</th>
      <th style="padding: 12px; text-align: left; border: 1px solid #444;">Description</th>
      <th style="padding: 12px; text-align: left; border: 1px solid #444;">Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 12px; border: 1px solid #444;"><code>learning_rate</code></td>
      <td style="padding: 12px; border: 1px solid #444;">Step size for gradient descent</td>
      <td style="padding: 12px; border: 1px solid #444;">0.1</td>
    </tr>
    <tr style="background-color: #1f1f1f;">
      <td style="padding: 12px; border: 1px solid #444;"><code>epochs</code></td>
      <td style="padding: 12px; border: 1px solid #444;">Maximum number of training iterations</td>
      <td style="padding: 12px; border: 1px solid #444;">10000</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #444;"><code>tol</code></td>
      <td style="padding: 12px; border: 1px solid #444;">Tolerance for early stopping</td>
      <td style="padding: 12px; border: 1px solid #444;">1e-5</td>
    </tr>
    <tr style="background-color: #1f1f1f;">
      <td style="padding: 12px; border: 1px solid #444;"><code>patience</code></td>
      <td style="padding: 12px; border: 1px solid #444;">Number of epochs to wait for improvement before stopping</td>
      <td style="padding: 12px; border: 1px solid #444;">20</td>
    </tr>
  </tbody>
</table>

---

## Usage

```python
import numpy as np
from linear_regression import LinearRegression

# Create and train model
lr = LinearRegression()

# Example data
X = np.random.uniform(0, 1, (100, 3))
true_w = np.array([2, 3, 4])
true_b = 6
y = np.dot(X, true_w) + true_b + np.random.randn(100) * 0.1  # with noise

# Train
w, b = lr.fit(
    x=X,
    y=y,
    learning_rate=0.1,
    epochs=10000,
    tol=1e-5,
    patience=20
)

print(f"Learned weights: {w}")
print(f"Learned bias: {b:.4f}")
