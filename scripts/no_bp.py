import chainer
import numpy as np
import chainer.computational_graph as c
import chainer.links as L
import chainer.functions as F


x = chainer.Variable(np.array([1, ], np.float32))
y1 = x ** 2
y2 = y1 ** 2
z1 = x ** 3
z2 = z1 ** 3
w = y2 + z2
w.backward()

print(w.grad)  # 1
print(z2.grad)  # None
print(z1.grad)  # None
print(y2.grad)  # None
print(y1.grad)  # None
print(x.grad)  # 13

g = c.build_computational_graph(w)
with open('with_bp', 'w') as o:
    o.write(g.dump())


# no_backprop mode
x = chainer.Variable(np.array([1, ], np.float32))
with chainer.no_backprop_mode():
    y1 = x ** 2
    y2 = y1 ** 2
z1 = x ** 3
z2 = z1 ** 3
w = y2 + z2
w.backward()

print(w.grad)  # 1
print(z2.grad)  # None
print(z1.grad)  # None
print(y2.grad)  # 1
print(y1.grad)  # None
print(x.grad)  # 9

g = c.build_computational_graph(w)
with open('without_bp', 'w') as o:
    o.write(g.dump())



# Chain?
class Model(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(1, 1)
    def __call__(self, x):
        x = self.l1(x)
        x = x ** 2
        return x

model = Model()
print(model.l1.W)

x = np.array([[1]], np.float32)
y = model(x)
z = model(x)
w = y + z
model.cleargrads()
w.backward(retain_grad=True)

print(w.grad)  # 1
print(z.grad)  # 1
print(y.grad)  # 1
print(model.l1.W.grad)  # 4 * c.l1.W

g = c.build_computational_graph(w)
with open('with_bp_chain', 'w') as o:
    o.write(g.dump())


# Chain? without bp
class Model(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(1, 1)
    def __call__(self, x):
        x = self.l1(x)
        x = x ** 2
        return x

model = Model()
print(model.l1.W)

x = np.array([[1]], np.float32)
with chainer.no_backprop_mode():
    y = model(x)
z = model(x)
w = y + z
model.cleargrads()
w.backward(retain_grad=True)

print(w.grad)  # 1
print(z.grad)  # 1
print(y.grad)  # 1
print(model.l1.W.grad)  # 4 * c.l1.W

g = c.build_computational_graph(w)
with open('without_bp_chain', 'w') as o:
    o.write(g.dump())
