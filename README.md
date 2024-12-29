# Linear Relation Between Two Features Using Gradient Descent

This repository demonstrates how to compute the linear relationship between two features using Gradient Descent. It provides an implementation in Python, along with visualization of the results.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Snippets](#code-snippets)
- [Results](#results)
- [Contributing](#contributing)

## Overview

Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning and deep learning. This repository provides a hands-on implementation of Gradient Descent to determine the linear relationship between two numerical features.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/venky0112/Linear-Relation-between-2-Features-using-Gradient-Descent.git

# Navigate to the project directory
cd Linear-Relation-between-2-Features-using-Gradient-Descent

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the notebook to experiment with Gradient Descent and visualize the results:

```bash
# Open the Jupyter Notebook
jupyter notebook gradient_descent.ipynb
```

Follow the instructions in the notebook to execute each cell and observe the output.

## Code Snippets

Below are some key snippets from the implementation:

### Data Initialization

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X = np.linspace(0, 10, 100)
y = 2.5 * X + np.random.randn(100) * 2

plt.scatter(X, y, color="blue")
plt.title("Generated Data")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.show()
```

### Gradient Descent Implementation

```python
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, b = 0, 0  # Initialize parameters
    n = len(y)

    for _ in range(epochs):
        y_pred = m * X + b
        dm = -(2/n) * sum(X * (y - y_pred))
        db = -(2/n) * sum(y - y_pred)

        m -= learning_rate * dm
        b -= learning_rate * db

    return m, b

# Run Gradient Descent
m, b = gradient_descent(X, y)
print(f"Slope (m): {m}, Intercept (b): {b}")
```

### Visualization of Results

```python
# Visualize the fitted line
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, m * X + b, color="red", label="Fitted Line")
plt.title("Linear Fit using Gradient Descent")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.legend()
plt.show()
```

## Results

By running the notebook, you will:

1. Generate a synthetic dataset.
2. Implement and apply Gradient Descent to find the optimal slope and intercept.
3. Visualize the linear relationship.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.
