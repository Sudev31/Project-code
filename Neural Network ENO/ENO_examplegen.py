#as we wanna do eno for degree = 2, let us implement lagrange interpolation for degree 2 only

#given {x_0,x_1,x_2} we want to interpolate a deg 2 polynomial through it

#if deg is fixed, we can define th elagrange polynomial directly
def set_plot_style():
    plt.rcParams['font.size'] = 24
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
import matplotlib.pyplot as plt
import numpy as np
X = [-2,-1.5, -1]
X = np.array(X)
Y = np.sin((X)**2)

#for i in range(3):
#    X.append(float(input()))
#    Y.append(float(input()))
p=2
n = 900
x = np.linspace(-2,-1,n)
y = np.zeros(n)
t = 1
for i in range(n):
  for j in range(len(X)):
    t =1
    for k in range(1,len(X)):
      t = t* (x[i] - X[(j+k)%(p+1)])/(X[j] - X[(j+k)%(p+1)])
    y[i] = y[i] + t*Y[j]
    t = 1



plt.figure(figsize=(10, 8))   
plt.plot(x, y, label="Interpolating Polynomial (Degree 2)", color="blue")
plt.scatter(X, Y, color="red", label="Data Points")  # Mark the given points
plt.title("Lagrange Interpolation (Degree 2)")
plt.xlabel("x")
plt.ylabel("y")

plt.legend()
plt.grid(True)
plt.savefig(f"ENO_Example2.pdf", bbox_inches='tight', dpi=300, transparent=True)
plt.show()


