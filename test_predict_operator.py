import deepxde as dde
import numpy as np
import torch

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, y):
    return dde.grad.jacobian(y, x, i=0, j=1) - dde.grad.jacobian(y, x, i=0, j=0)

data = dde.data.TimePDE(geomtime, pde, [], num_domain=10)
net = dde.nn.FNN([2, 10, 1], "tanh", "Glorot normal")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

X = np.array([[0.0, 0.5], [0.5, 0.5]])
y = model.predict(X)
print("y:", y)

def dt_operator(x, y, X_tensor):
    return dde.grad.jacobian(y, x, i=0, j=1)

y_t = model.predict(X, operator=dt_operator)
print("y_t:", y_t)
