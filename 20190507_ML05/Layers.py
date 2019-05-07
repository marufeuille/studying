from Functions import softmax, cross_entropy_error
import switchCupy
xp = switchCupy.xp_factory()

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Affine:
    def __init__(self, W, b):
        self.W  = W
        self.b  = b
        self.x  = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = xp.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = xp.dot(dout, self.W.T)
        self.dW = xp.dot(self.x.T, dout)
        self.db = xp.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y    = None
        self.t    = None

    def forward(self, x, t):
        self.t    = t
        self.y    = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


