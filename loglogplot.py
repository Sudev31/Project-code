import numpy as np
import matplotlib.pyplot as plt
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys

# Read the convergence data
data = np.loadtxt("convergence_table.txt", skiprows=1)  # Skip header
time = data[:, 2]  # Third column is CPU time
error = data[:, 3]  # Fourth column is L1 error

data1 = np.loadtxt("convergence_PINN_table.txt", skiprows=1)
time1 = data1[:, 2]  # Third column is CPU time
error1 = data1[:, 3]  # Fourth column is L1 error
# Create log-log plot
plt.figure(figsize=(8, 6))
plt.loglog(time, error, 'o-', label='Upwind Scheme', markersize=8, linewidth=2)
plt.loglog(time1, error1,'x-', label='PINN Scheme', markersize=8, linewidth=2)
# Add a reference line for slope (e.g., slope = -1 for ideal scaling)
x_fit = np.linspace(min(time), max(time), 100)


# Labels and title
plt.xlabel("CPU Time (s) [log scale]", fontsize=14)
plt.ylabel("L1 Error [log scale]", fontsize=14)
plt.title("Log-Log Plot of CPU Time vs. L1 Error", fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.savefig("time_vs_error.pdf", dpi=300, bbox_inches='tight')
plt.show()