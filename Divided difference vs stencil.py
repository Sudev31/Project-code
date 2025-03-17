#as we want to do eno for degree = 2, let us implement ENO with 0 shift interpolation for degree 2 only

#given {x_0,x_1,x_2} we want to interpolate a deg 2 polynomial through it

import matplotlib.pyplot as plt
import numpy as np
X_main = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
Y_main = [2, 1.5, 1, 0.5, 0, 0.5, 1, 1.5, 2]
#Y_main = [4, 2.25, 1, 0.25, 0, 0.25, 1, 2.25, 4]
#X_main = np.array(X_main)
#Y_main = np.sin((X_main)**2)
p = 3 #deg+1 in here
n = p+(p-2)
#for i in range(3):
#    X.append(float(input()))
#    Y.append(float(input()))

#update r formula in the other code previous one
r=np.zeros(len(X_main)-2*p+3, dtype=int)
d = np.zeros((2*(len(X_main)-2*p+3)))
for k in range(len(X_main)-2*p+3):


  X = X_main[k:k+(p+(p-2))]
  Y = Y_main[k:k+(p+(p-2))]
  D = np.zeros((p, n))

  D[0] = Y

  for j in range(1, p): 
    D[j, :(n-j)] = D[j-1, 1:n-(j-1)] - D[j-1, :n-j] 


  for l in range(2,p):
    d[2*k] = abs(D[l][p-3-r[k]])
    d[2*k+1] = abs(D[l][p-2-r[k]])
    if (abs(D[l][p-3-r[k]])<abs(D[l][p-2-r[k]])):
      r[k]+=1


n = 900
p=p-1
y = np.zeros(n)
t = 1
for l in range(len(X_main)-2*(p-1)-1):      #no. of intervals
  x = np.linspace(X_main[(p-1)+l],X_main[p+l],n)
  X = X_main[l+(p-1)-r[l]:l+2*p-r[l]]  
  Y = Y_main[l+(p-1)-r[l]:l+2*p-r[l]]
  y = np.zeros(n)
  for i in range(n):
    for j in range(len(X)):
      t =1
      for k in range(1,len(X)):
        t = t* (x[i] - X[(j+k)%(p+1)])/(X[j] - X[(j+k)%(p+1)])
      y[i] = y[i] + t*Y[j]
      t = 1
  plt.plot(x, y, color="blue")
  plt.scatter(X, Y, color="red")  
  plt.title("ENO Interpolation (Degree 2) for f(x) = sgn(x)")
  plt.xlabel("x")
  plt.ylabel("y")







plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np


p = 2  # Degree + 1
n = 900
r_values = 2

for l in range(len(X_main) - 2 * (p - 1) - 1):
    for r in range(r_values):
        x = np.linspace(X_main[(p - 1) + l], X_main[p + l], n)
        X = X_main[l + (p - 1) - r:l + 2 * p - r]
        Y = Y_main[l + (p - 1) - r:l + 2 * p - r]
        y = np.zeros(n)
        for i in range(n):
            for j in range(len(X)):
                t = 1
                for k in range(1, len(X)):
                    t = t * (x[i] - X[(j + k) % (p + 1)]) / (X[j] - X[(j + k) % (p + 1)])
                y[i] += t * Y[j]

        # Plotting the graph
        print(l)
        if r == 0:
            plt.plot(x, y, color="green")
            
            plt.text(x[n//2], y[n//2], f"{d[2*l+1]:.2f}", color="green", fontsize=10)
        else:
            plt.plot(x, y, color="blue")
            
            plt.text(x[n//3], y[n//3], f"{d[2*l]:.2f}", color="blue", fontsize=10)


        plt.scatter(X, Y, color="red")  # Mark the given points

plt.title("Interpolation (Degree 2)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
print(d)



