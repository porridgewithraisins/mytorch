import tensa as t


@t.fuse
def sigmoid(x):
    out = 1 / (1 + t.exp(-x))

    def sigmoid_backward(d):
        return d * out * (1 - out)

    return x, out, sigmoid_backward


@t.fuse
def tanh(x):
    out = t.np.tanh(x.data)

    def tanh_backward(d):
        return d * (1 - out**2)

    return x, out, tanh_backward


def relu(x):
    return t.where(x.data > 0, x, 0)


@t.fuse
def softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    e = t.exp(shifted)
    out = e / e.sum(axis=axis, keepdims=True)

    def softmax_backward(d):
        return out * (d - (d * out).sum(axis=axis, keepdims=True))

    return x, out, softmax_backward


@t.fuse
def log_softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    log_z = t.log(t.exp(x - x_max).sum(axis=axis, keepdims=True)) + x_max
    out = x - log_z

    def log_softmax_backward(d):
        return d - t.exp(out) * d.sum(axis=axis, keepdims=True)

    return x, out, log_softmax_backward
