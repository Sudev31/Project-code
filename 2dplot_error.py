"""
plot_error_from_plt.py

Read a PLT-like CSV file with a 3-line header (third line contains SOLUTIONTIME),
compute the pointwise absolute error against a known analytical solution, and
save a filled-contour plot of the error.

Expected input file format (after 3 header lines):
    x,y,value
    x,y,value
    ...

Usage:
    python plot_error_from_plt.py path/to/solution.plt figures/error_t.png

Notes:
- The analytical solution used here is u_e(x,y,t) = sin(2π(x - t)) * sin(2π(y - t)).
  Make sure the numerical solution in the file uses the same normalization/units.
- The script grids scattered points by unique x and y values; it assumes the
  data lives on a rectilinear grid (same set of x values for each y row).
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys

def u_e(x, y, t):
    return np.sin(2*np.pi * (x - t)) * np.sin(2*np.pi * (y - t))

def get_solution_time(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # The third line (index 2) contains SOLUTIONTIME
        third_line = lines[2].strip()
        # Split into parts using commas
        parts = third_line.split(', ')
        for part in parts:
            if part.startswith('SOLUTIONTIME='):
                time_str = part.split('=')[1]
                return float(time_str)
        # If not found, raise error
        raise ValueError("SOLUTIONTIME not found in the third line of the PLT file.")

if len(sys.argv) != 3:
    print("Usage: python mycode.py <file_path> <figure_name>")
    sys.exit(1)

file_path = sys.argv[1]
figure_name = sys.argv[2]

# Read the solution time from the PLT file
t = get_solution_time(file_path)

# Load the data from the text file, skipping the first 3 lines
data = np.loadtxt(file_path, delimiter=',', skiprows=3)

# Split the data into coordinates (first two columns) and solution (third column)
coordinates = data[:, :2]  # x, y
solution = data[:, 2]      # numerical solution

x = coordinates[:, 0]
y = coordinates[:, 1]
z_numerical = solution

# Compute exact solution and error
z_exact = u_e(x, y, t)
error = np.abs(z_exact - z_numerical)

# Identify the unique x and y coordinates to form the grid
unique_x = np.unique(x)
unique_y = np.unique(y)
grid_shape = (len(unique_y), len(unique_x))

# Initialize the error matrix
error_matrix = np.full(grid_shape, np.nan)

# Populate the error matrix
for i in range(len(x)):
    x_idx = np.where(unique_x == x[i])[0][0]
    y_idx = np.where(unique_y == y[i])[0][0]
    error_matrix[y_idx, x_idx] = error[i]

# Create meshgrid for plotting
X, Y = np.meshgrid(unique_x, unique_y)

def set_plot_style():
    plt.rcParams['font.size'] = 24
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
set_plot_style()
# Plotting
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, error_matrix, levels=20)
plt.colorbar(contour, orientation='vertical', pad=0.07, shrink=0.58, aspect=40)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f"Error at t={t:.2f}")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(False)

plt.savefig(figure_name, bbox_inches='tight', dpi=300, transparent=True)
