import functools
from contextlib import contextmanager
import numpy as np

_no_grad_depth = 0


def _is_grad_disabled():
    return _no_grad_depth > 0


@contextmanager
def no_grad():
    global _no_grad_depth
    _no_grad_depth += 1
    try:
        yield
    finally:
        _no_grad_depth -= 1


# User-visible tensor constructor
def tensor(data=None, shape=(), requires_grad=False):
    if isinstance(data, Tensor):
        return data
    data = np.empty(shape) if data is None else np.asarray(data)
    if shape:
        data = data.reshape(shape)
    requires_grad = requires_grad and not _is_grad_disabled()
    return Tensor(data, requires_grad=requires_grad, parents=(), grad_fn=None)


class Tensor:
    def __init__(self, data, requires_grad=False, parents=(), grad_fn=None):
        self.data = data
        self.requires_grad = requires_grad and not _is_grad_disabled()
        self.grad = None
        if self.requires_grad:
            self.parents = parents
            self.grad_fn = grad_fn

    def backward(self, grad=None, higher_order=False):
        if not self.requires_grad or _is_grad_disabled():
            return
        topo = []
        visited = set()

        stack = [self]
        while stack:
            node = stack[-1]
            if id(node) not in visited:
                visited.add(id(node))
                stack.extend(filter(lambda p: p.requires_grad, node.parents))
            else:
                stack.pop()
                if not _is_leaf(node):
                    topo.append(node)

        self.grad = _to_tensor(grad if grad is not None else 1.0)
        for node in reversed(topo):
            grad_fn = node.grad_fn if higher_order else _with_no_grad(node.grad_fn)
            grads = grad_fn(node.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for parent, g in zip(node.parents, grads):
                g = _to_tensor(g)
                if parent.requires_grad:
                    if parent.grad is None:
                        parent.grad = g
                    else:
                        parent.grad = parent.grad + g

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    def item(self):
        return self.data.item()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __neg__(self):
        def neg_backward(d):
            return -d

        return Tensor(-self.data, requires_grad=self.requires_grad, parents=(self,), grad_fn=neg_backward)

    def __add__(self, other):
        other = _to_tensor(other)

        def add_backward(d):
            return _bg(d, self.data.shape), _bg(d, other.data.shape)

        return Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            grad_fn=add_backward,
        )

    def __mul__(self, other):
        other = _to_tensor(other)

        def mul_backward(d):
            return _bg(d * other, self.data.shape), _bg(d * self, other.data.shape)

        return Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            grad_fn=mul_backward,
        )

    def __sub__(self, other):
        other = _to_tensor(other)

        def sub_backward(d):
            return _bg(d, self.data.shape), _bg(-d, other.data.shape)

        return Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            grad_fn=sub_backward,
        )

    def __truediv__(self, other):
        other = _to_tensor(other)
        safe = np.where(other.data == 0, 1e-8, other.data)

        def truediv_backward(d):
            return _bg(d / safe, self.shape), _bg(-d * self / safe**2, other.shape)

        return Tensor(
            self.data / safe,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            grad_fn=truediv_backward,
        )

    def __iadd__(self, other):
        if self.requires_grad:
            raise RuntimeError("in-place operations on tensors with requires_grad=True are not supported")
        self.data += _to_tensor(other).data
        return self

    def __isub__(self, other):
        if self.requires_grad:
            raise RuntimeError("in-place operations on tensors with requires_grad=True are not supported")
        self.data -= _to_tensor(other).data
        return self

    def __eq__(self, other):
        return Tensor(self.data == _to_tensor(other).data, requires_grad=False)

    def __ne__(self, other):
        return Tensor(self.data != _to_tensor(other).data, requires_grad=False)

    def __lt__(self, other):
        return Tensor(self.data < _to_tensor(other).data, requires_grad=False)

    def __gt__(self, other):
        return Tensor(self.data > _to_tensor(other).data, requires_grad=False)

    def __le__(self, other):
        return Tensor(self.data <= _to_tensor(other).data, requires_grad=False)

    def __ge__(self, other):
        return Tensor(self.data >= _to_tensor(other).data, requires_grad=False)

    def __invert__(self):
        return Tensor(~self.data, requires_grad=False)

    def __matmul__(self, other):
        other = _to_tensor(other)

        def matmul_backward(d):
            if self.ndim == 1 and other.ndim == 1:
                return d * other, d * self
            d = d.unsqueeze(1) if d.ndim == 1 else d
            a = self.unsqueeze(1) if self.ndim == 1 else self
            b = other.unsqueeze(1) if other.ndim == 1 else other
            da = d @ b.swapaxes(-1, -2)
            db = a.swapaxes(-1, -2) @ d
            if self.ndim == 1:
                da = da.squeeze()
            if other.ndim == 1:
                db = db.squeeze()
            return da, db

        return Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            grad_fn=matmul_backward,
        )

    def __pow__(self, other):
        other = _to_tensor(other)
        assert other.data.ndim == 0, "__pow__ expects scalar exponent"
        n = other.data.item()

        def pow_backward(d):
            return d * n * (self ** (n - 1))

        return Tensor(
            self.data**n, requires_grad=self.requires_grad or other.requires_grad, parents=(self,), grad_fn=pow_backward
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return _to_tensor(other).__sub__(self)

    def __rtruediv__(self, other):
        return _to_tensor(other).__truediv__(self)

    def __rmatmul__(self, other):
        return _to_tensor(other).__matmul__(self)

    def sum(self, axis=None, keepdims=False):
        return _reduce(self, np.sum, lambda t, out: t * 0 + 1, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        return _reduce(self, np.max, lambda t, out: t == out, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return _reduce(self, np.min, lambda t, out: t == out, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        n = self.data.shape[axis] if axis is not None else self.data.size
        return self.sum(axis=axis, keepdims=keepdims) / n

    def transpose(self, axes=None):
        """`axes` argument behaves the same way as in `np.transpose`"""

        def transpose_backward(d):
            return d.transpose(np.argsort(axes)) if axes is not None else d.T

        return Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            parents=(self,),
            grad_fn=transpose_backward,
        )

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        def swapaxes_backward(d):
            return d.swapaxes(axis1, axis2)

        return Tensor(
            np.swapaxes(self.data, axis1, axis2),
            requires_grad=self.requires_grad,
            parents=(self,),
            grad_fn=swapaxes_backward,
        )

    def reshape(self, *shape):
        original_shape = self.data.shape

        def reshape_backward(d):
            return d.reshape(original_shape)

        return Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            parents=(self,),
            grad_fn=reshape_backward,
        )

    def squeeze(self, axis=None):
        original_shape = self.data.shape

        def squeeze_backward(d):
            return d.reshape(original_shape)

        return Tensor(
            np.squeeze(self.data, axis=axis),
            requires_grad=self.requires_grad,
            parents=(self,),
            grad_fn=squeeze_backward,
        )

    def unsqueeze(self, axis):
        def unsqueeze_backward(d):
            return d.squeeze(axis=axis)

        return Tensor(
            np.expand_dims(self.data, axis=axis),
            requires_grad=self.requires_grad,
            parents=(self,),
            grad_fn=unsqueeze_backward,
        )

    def __getitem__(self, idx):

        def getitem_backward(d):
            g = self * 0
            return g.scatter(idx, d)

        return Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            parents=(self,),
            grad_fn=getitem_backward,
        )

    def scatter(self, idx, source):
        out = self.data.copy()
        src = source.data if isinstance(source, Tensor) else source
        np.add.at(out, idx, src)

        def scatter_backward(d):
            return d, d[idx]

        return Tensor(
            out,
            requires_grad=self.requires_grad or (isinstance(source, Tensor) and source.requires_grad),
            parents=(self, source) if isinstance(source, Tensor) else (self,),
            grad_fn=scatter_backward,
        )

    def __setitem__(self, idx, value):
        if self.requires_grad:
            raise RuntimeError(
                "in-place operations on tensors with requires_grad=True are not supported — they break the computation graph"
            )
        self.data[idx] = value


def exp(t):
    out = np.exp(t.data)

    def exp_backward(d):
        return d * out

    return Tensor(out, requires_grad=t.requires_grad, parents=(t,), grad_fn=exp_backward)


def log(t):
    safe = np.where(t.data <= 0, 1e-8, t.data)

    def log_backward(d):
        return d / safe

    return Tensor(np.log(safe), requires_grad=t.requires_grad, parents=(t,), grad_fn=log_backward)


def sqrt(t):
    safe = np.where(t.data <= 0, 1e-8, t.data)
    out = np.sqrt(safe)

    def sqrt_backward(d):
        return d / (2 * out)

    return Tensor(out, requires_grad=t.requires_grad, parents=(t,), grad_fn=sqrt_backward)


def abs(t):
    def abs_backward(d):
        return d * np.sign(t.data)

    return Tensor(np.abs(t.data), requires_grad=t.requires_grad, parents=(t,), grad_fn=abs_backward)


def cat(ts, axis=0):
    data = [t.data for t in ts]
    splits = np.cumsum([d.shape[axis] for d in data[:-1]])

    def cat_backward(d):
        return tuple(_to_tensor(g) for g in np.split(d.data, splits, axis=axis))

    return Tensor(
        np.concatenate(data, axis=axis),
        requires_grad=any(t.requires_grad for t in ts),
        parents=tuple(ts),
        grad_fn=cat_backward,
    )


def where(condition, x, y):
    x, y = _to_tensor(x), _to_tensor(y)

    def where_backward(d):
        return _bg(d * condition, x.shape), _bg(d * ~condition, y.shape)

    return Tensor(
        np.where(condition.data, x.data, y.data),
        requires_grad=x.requires_grad or y.requires_grad,
        parents=(x, y),
        grad_fn=where_backward,
    )


def fuse(fn):
    def wrapper(*args, **kwargs):
        inputs, out, grad_fn = _with_no_grad(fn)(*args, **kwargs)
        parents = inputs if isinstance(inputs, tuple) else (inputs,)
        return Tensor(
            out.data,
            requires_grad=any(p.requires_grad for p in parents),
            parents=parents,
            grad_fn=grad_fn,
        )

    return wrapper


# Internal


def _is_leaf(t):
    return len(t.parents) == 0


def _bg(g, original_shape):
    while g.ndim > len(original_shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(original_shape):
        if size == 1:
            g = g.sum(axis=axis, keepdims=True)
    return g


def _to_tensor(x):
    if isinstance(x, Tensor):
        return x

    return Tensor(np.asarray(x), requires_grad=False, parents=(), grad_fn=None)


def _reduce(x, np_fn, mask_fn, axis=None, keepdims=False):
    out = np_fn(x.data, axis=axis, keepdims=keepdims)

    def grad_fn(d):
        out_expanded = out if keepdims or axis is None else np.expand_dims(out, axis=axis)
        d_expanded = d if keepdims or axis is None else d.unsqueeze(axis)
        return mask_fn(x, _to_tensor(out_expanded)) * d_expanded

    return Tensor(out, requires_grad=x.requires_grad, parents=(x,), grad_fn=grad_fn)


def _with_no_grad(fn):
    if fn is None:
        return None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _no_grad_depth
        _no_grad_depth += 1
        result = fn(*args, **kwargs)
        _no_grad_depth -= 1
        return result

    return wrapper
