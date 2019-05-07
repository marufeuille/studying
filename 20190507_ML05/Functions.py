import switchCupy
xp = switchCupy.xp_factory()

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - xp.max(x, axis=0)
        y = xp.exp(x) / xp.sum(xp.exp(x), axis=0)
        return y.T

    x = x - xp.max(x)
    return xp.exp(x) / xp.sum(xp.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -xp.sum(xp.log(y[xp.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4
    grad = xp.zeros_like(x)

    it = xp.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext()

    return grad
