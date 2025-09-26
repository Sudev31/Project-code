"""
Brief:
    Read a PLT-like CSV file with a header that contains SOLUTIONTIME and
    scatter (x,y,value) rows. Produce a filled contour plot (contourf)
    with a title that includes the solution time.

Usage:
    python 2dplot_new.py path/to/solution.plt figures/solution_t.png
"""

"""
Differences from 2plot.py
-------------------------

This script (`2dplot_new.py`) is an enhanced/production-ready version of the
quick plotting script (`2plot.py`). Key differences and improvements:

1. SOLUTIONTIME parsing
   - `2dplot_new.py` reads the SOLUTIONTIME value from the 3rd header line of the
     input PLT file and prints/displays it in the plot title (useful for time-series).
   - `2plot.py` does not extract or show time information.

2. Plot type
   - `2dplot_new.py` uses `contourf` (filled contours) for smoother, publication-ready
     color regions.
   - `2plot.py` uses `contour` (line contours), which is faster for quick checks.

3. Styling and presentation
   - `2dplot_new.py` includes `set_plot_style()` to increase font sizes, line widths,
     and use a serif font for higher-quality figures suitable for reports/overleaf.
   - Colorbar orientation: `2dplot_new.py` uses a vertical colorbar (better for tall plots),
     while `2plot.py` uses a horizontal one.

4. Title & annotation
   - `2dplot_new.py` shows the time in the title (e.g., "Solution at t=0.25"), making it
     easier to compare snapshots from different timesteps.
   - `2plot.py` uses a generic title "Solution".

5. Use case
   - `2dplot_new.py` is intended for final figures and report images (presentation/paper).
   - `2plot.py` is intended for quick diagnostic visualization during development.

Notes:
- Both scripts share the same core logic (gridification of scattered x,y,value data).
- If you plan to maintain both, keep `2plot.py` as a fast debug tool and `2dplot_new.py`
  for producing final images for the README/Overleaf.
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys

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

# run like $python plot1.py sol.plt sol.pdf
# Check if the user provided the file path as a command-line argument
if len(sys.argv) != 3:
    print("Usage: python mycode.py <file_path>")
    sys.exit(1)

# Get the file path from the command-line argument
file_path = sys.argv[1]
figure_name = sys.argv[2]
t = get_solution_time(file_path)
# Load the data from the text file, skipping the first 3 lines
data = np.loadtxt(file_path, delimiter=',', skiprows=3)

# Split the data into coordinates (first two columns) and solution (third column)
coordinates = data[:,:2]  # First two columns (x, y)
solution = data[:,2]      # Third column (solution)

# x and y are the coordinates from coordinates1
x = coordinates[:, 0]
y = coordinates[:, 1]
z = solution  # The solution values

# Identify the unique x and y coordinates  form the grid
unique_x = np.unique(x)
unique_y = np.unique(y)

# Determine the grid shape (number of unique points in x and y directions)
grid_shape = (len(unique_y), len(unique_x))

# Reshape x, y, and z into a grid form (matrix form) based on the unique x and y values
x_grid = unique_x
y_grid = unique_y

# Initialize the z matrix (solution) with the same shape as the grid
z_matrix = np.full(grid_shape, np.nan)

# Populate the z_matrix with the corresponding z values from the scattered data
for i in range(len(x)):
    # Find the index of x and y in the grid
    x_idx = np.where(unique_x == x[i])[0][0]
    y_idx = np.where(unique_y == y[i])[0][0]
    
    # Assign the corresponding z value to the matrix
    z_matrix[y_idx, x_idx] = z[i]
# Now x_grid, y_grid, and z_matrix represent the data in grid and matrix form

# Create a meshgrid for the x and y coordinates
X, Y = np.meshgrid(x_grid, y_grid)

def set_plot_style():
    plt.rcParams['font.size'] = 24
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24

set_plot_style()
# Plot the pseudocolor plot
plt.figure(figsize=(8, 6))
pcolor_plot = plt.contourf(X, Y, z_matrix, levels = 20)

# Add a color bar to indicate the solution values
# change apsect to more to make it thinner
cbar = plt.colorbar(pcolor_plot, orientation='vertical', pad=0.07, shrink=0.58, aspect = 40 )
#cbar.set_label('Solution')

# Set plot labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f"Solution at t={t:.2f}")

# Set aspect ratio to preserve the data's scale
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(False)
# Display the plot
#plt.show()
plt.savefig(figure_name, bbox_inches='tight', dpi=300, transparent=True)
