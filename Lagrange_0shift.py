#as we want to do eno for degree = 2, let us implement lagrange interpolation for degree 2 only

#given {x_0,x_1,x_2} we want to interpolate a deg 2 polynomial through it

#if deg is fixed, we can define th elagrange polynomial directly
import matplotlib.pyplot as plt
import numpy as np

s = 500
X_main = np.linspace(-1,1,s)
Y_main=np.zeros(s)
for i in range(len(X_main)):
  if (X_main[i]>0.5):
    Y_main[i]=1
  elif (X_main[i]<-0.5):
    Y_main[i]=-1
  elif(X_main[i] == 0):
    Y_main[i]=0
  else:
    Y_main[i] = np.sin(1/X_main[i])


p = 3
 #deg+1 in here
n = p+(p-2)
#for i in range(3):
#    X.append(float(input()))
#    Y.append(float(input()))

#update r formula in the other code previous one
r=np.zeros(len(X_main)-2*p+3, dtype=int)

for k in range(len(X_main)-2*p+3):


  X = X_main[k:k+(p+(p-2))]
  Y = Y_main[k:k+(p+(p-2))]
  D = np.zeros((p, n))
  D[0] = Y

  for j in range(1, p): # Changed loop range to start from 1
    D[j, :(n-j)] = D[j-1, 1:n-(j-1)] - D[j-1, :n-j] # Subtraction from the second element onward

  for l in range(2,p):
    if (abs(D[l][p-3-r[k]])<abs(D[l][p-2-r[k]])):
      r[k]+=1
print(D)
print(r)

n = 900
p=p-1
y = np.zeros(n)
t = 1
for l in range(len(X_main)-2*(p-1)-1):
  x = np.linspace(X_main[(p-1)+l],X_main[p+l],n)
  X = X_main[l+(p-1)-r[l]:l+2*p-r[l]]  #[2:5]  [1:4]
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
  plt.scatter(X, Y, color="red")  # Mark the given points
  plt.title("Lagrange Interpolation (Degree 2)")
  plt.xlabel("x")
  plt.ylabel("y")







plt.legend()
plt.grid(True)
plt.show()




