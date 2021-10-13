#Zadanie 1
import numpy as np
A = np.random.normal(size=(9,10))
B = np.random.normal(size=(10,6))
C = np.matmul(A,B)
assert A.shape != B.shape
assert C.shape == (9,6)

#Zadanie 2
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
X = np.random.uniform(low=-10, high=10, size=(1000, 2000))

assert sigmoid(X).max() <= 1.
assert sigmoid(X).min() >= 0.
#Zadanie 3
X = np.random.uniform(low=-10, high=10, size=(100, 10))
print(X)

X_hat = (X - X.mean(axis=0))/X.std(axis=0)
assert np.allclose(X_hat.mean(0), 0.)
assert np.allclose(X_hat.std(0), 1.)

