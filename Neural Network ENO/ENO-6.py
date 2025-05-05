import numpy as np



# for building and training neural networks
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.regularizers import l2

import numpy as np

#let's us try with the function f(x) = x^2+sin(x)
#also the case being implemented here is for p=3 only, for higher p , the last if statement will also havea for loop.
p = 6
s = 2000

x = np.linspace(-1,1,s)
n = p+(p-2)
D = np.zeros((p, p+(p-2)))
Y=np.zeros(p+(p-2))
X_train = []
r=np.zeros(s-9, dtype=int)
for k in range(s-9):
  X = x[k:k+10]  #here we are taking 6 points one by one
  for i in range(10):
    if (-0.5<X[i]<0.5):
      Y[i]=-1
    elif (X[i]<-0.5):
      Y[i]=1
    elif(X[i] == 0):
      Y[i]=0
    else:
      Y[i] = np.sin(1/X[i])
  X_train.append(Y.copy())
  D = np.zeros((p, n))
  D[0] = Y


  for j in range(1, p): # Changed loop range to start from 1
    D[j, :(n-j)] = D[j-1, 1:n-(j-1)] - D[j-1, :n-j] # Subtraction from the second element onward

  for l in range(2,p):
    if (abs(D[l][p-3-r[k]])<abs(D[l][p-2-r[k]])):
      r[k]+=1
print(D)
print(r)
X_train = np.array(X_train)
X_train = np.array(X_train)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train_normalized = (X_train - mean) / std
X_train = X_train_normalized


from tensorflow.keras import regularizers # Import the regularizers module

# Assuming X_train and r are already defined properly
# For example, X_train should be of shape (num_samples, 6) and r contains integer labels in [0, p-1]

model = Sequential([
    tf.keras.Input(shape=(10,)),  # Input size for stencil points
    Dense(28, activation='relu'),  # Hidden layer 1 with L2 regularization
    Dense(20, activation='relu'),   # Hidden layer 2
    Dense(40, activation='relu'),
    Dense(40, activation='relu'), # Hidden layer 3
    Dense(10, activation='relu'),  # Hidden layer 1 with L2 regularization

    Dense(5, activation='softmax') # Output layer for multiclass classification
])

# Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Use sparse labels
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)
e =200
# Train the model
history = model.fit(
    X_train, r,  # Sparse labels (integer form)
    epochs=e,
    verbose=1
)

training_mse = history.history['loss']
plt.plot(range(1, e+1), training_mse, label='Training MSE')

plt.show()

"""
Here I am using the above model to predict the p=6 stencil shift for other functions, namely heaviside function and mod(x) function. 


s = 600
X_main = np.linspace(-1, 1, s) 
Y = np.zeros(10)  # Single Y array (persistent across loop)
predictions = np.zeros(s - 9)


for k in range(s-9):  
    X = X_main[k:k+10]  
    for i in range(10):
        if X[i] > 0:
            Y[i] = 1
        elif X[i] < 0:
            Y[i] = -1
        else:
            Y[i] = 1
    X_test = Y.reshape(1, -1)
    X_test_normalized = (X_test - mean) / std
    predictions[k] = np.argmax(model.predict(X_test))  # Printing Y in both iterations
print(predictions)

s = 600
X_main = np.linspace(-1,1,s)
Y_main = np.sin(X_main**2)

for i in range(len(X_main)):
  if X_main[i]>0:
    Y_main[i]=1
  elif X_main[i]<0:
    Y_main[i]=-1
  else:
    Y_main[i] = 1  
  

p = 6
 #deg+1 in here
n = p+(p-2)
#for i in range(3):
#    X.append(float(input()))
#    Y.append(float(input()))

#update r formula in the other code previous one
r = predictions
r = np.round(predictions).astype(int)

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
  plt.title("ENO Interpolation (Degree 2)")
  plt.xlabel("x")
  plt.ylabel("y")




plt.legend()
plt.grid(True)
plt.show()




import numpy as np

s = 60
X_main = np.linspace(-1, 1, s) 
Y = np.zeros(10)  # Single Y array (persistent across loop)
predictions = np.zeros(s - 9)


for k in range(s-9):  
    X = X_main[k:k+10]  
    for i in range(10):
        if X[i] > 0:
            Y[i] = X[i]
        elif X[i] < 0:
            Y[i] = -X[i]
        else:
            Y[i] = 0
    X_test = Y.reshape(1, -1)
    X_test_normalized = (X_test - mean) / std
    predictions[k] = np.argmax(model.predict(X_test))  # Printing Y in both iterations
print(predictions)


import matplotlib.pyplot as plt
import numpy as np
s = 60
X_main = np.linspace(-1,1,s)
Y_main = np.sin(X_main**2)

for i in range(len(X_main)):
  if X_main[i]>0:
    Y_main[i]=X_main[i]
  elif X_main[i]<0:
    Y_main[i]=-X_main[i]
  else:
    Y_main[i] = 0 
  

p = 6
 #deg+1 in here
n = p+(p-2)
#for i in range(3):
#    X.append(float(input()))
#    Y.append(float(input()))

#update r formula in the other code previous one
r = predictions
r = np.round(predictions).astype(int)

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
  plt.title("ENO Interpolation (Degree 2)")
  plt.xlabel("x")
  plt.ylabel("y")




plt.legend()
plt.grid(True)
plt.show()
"""