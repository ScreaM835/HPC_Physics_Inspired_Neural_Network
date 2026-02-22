import deepxde as dde
import numpy as np
import torch

geom = dde.geometry.Interval(-1, 1)
timedomain1 = dde.geometry.TimeDomain(0, 1)
geomtime1 = dde.geometry.GeometryXTime(geom, timedomain1)

def pde(x, y):
    return dde.grad.jacobian(y, x, i=0, j=1) - dde.grad.jacobian(y, x, i=0, j=0)

data1 = dde.data.TimePDE(geomtime1, pde, [], num_domain=10)
net = dde.nn.FNN([2, 10, 1], "tanh", "Glorot normal")
model1 = dde.Model(data1, net)
model1.compile("adam", lr=1e-3)
model1.train(iterations=10)

print("Model 1 trained.")

timedomain2 = dde.geometry.TimeDomain(0, 2)
geomtime2 = dde.geometry.GeometryXTime(geom, timedomain2)
data2 = dde.data.TimePDE(geomtime2, pde, [], num_domain=20)
model2 = dde.Model(data2, net)
model2.compile("adam", lr=1e-3)
model2.train(iterations=10)

print("Model 2 trained with same net.")
