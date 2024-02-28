import numpy as np
from scipy.interpolate import interpn

# Sample 2D Grid Coordinates
x = np.linspace(0, 5, 4)
y = np.linspace(0, 3, 3)

# Sample Height Values (make up any data)
def f(x, y):
    return np.sin(x * y) * np.cos(2 * x)

values = f(x[:, np.newaxis], y)

# Interpolation Points 
interp_points = np.array([[1.2, 2.5],
                          [3.8, 1.7]])

# Linear Interpolation
interp_values = interpn(points=(x, y), values=values, xi=interp_points, method='linear')
print("Interpolated values:", interp_values)
