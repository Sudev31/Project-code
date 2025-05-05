import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys

# run like $python plot1.py sol.plt sol.pdf
# Check if the user provided the file path as a command-line argument
if len(sys.argv) != 3:
    print("Usage: python mycode.py <file_path>")
    sys.exit(1)

# Get the file path from the command-line argument
file_path = sys.argv[1]
figure_name = sys.argv[2]

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

# Plot the pseudocolor plot
plt.figure(figsize=(8, 6))
pcolor_plot = plt.contour(X, Y, z_matrix, levels = 20)

# Add a color bar to indicate the solution values
# change apsect to more to make it thinner
cbar = plt.colorbar(pcolor_plot, orientation='horizontal', pad=0.07, shrink=0.58, aspect = 40 )
#cbar.set_label('Solution')

# Set plot labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Solution')

# Set aspect ratio to preserve the data's scale
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(False)
# Display the plot
#plt.show()
plt.savefig(figure_name, bbox_inches='tight', dpi=300, transparent=True)