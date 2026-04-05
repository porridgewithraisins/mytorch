class Tensor:
    def __init__(self, data, requires_grad=False, ndim=None):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.ndim = ndim

    def __repr__(self):
        return f"Tensor({self.data})"

    def __neg__(self):
        bwd = lambda g: (-g,)
        return Tensor(("neg", self, bwd), requires_grad=self.requires_grad, ndim=self.ndim)

    def __add__(self, other):
        other = _to_tensor(other)
        bwd = lambda g: (g, g)
        return Tensor(
            ("add", self, other, bwd),
            requires_grad=self.requires_grad or other.requires_grad,
            ndim=max(self.ndim, other.ndim),
        )

    def __radd__(self, other):
        return self + _to_tensor(other)

    def __mul__(self, other):
        other = _to_tensor(other)
        bwd = lambda g: (g * other, g * self)
        return Tensor(
            ("mul", self, other, bwd),
            requires_grad=self.requires_grad or other.requires_grad,
            ndim=max(self.ndim, other.ndim),
        )

    def __rmul__(self, other):
        return self * _to_tensor(other)

    def __sub__(self, other):
        other = _to_tensor(other)
        bwd = lambda g: (g, -g)
        return Tensor(
            ("sub", self, other, bwd),
            requires_grad=self.requires_grad or other.requires_grad,
            ndim=max(self.ndim, other.ndim),
        )

    def __rsub__(self, other):
        return _to_tensor(other) - self

    def __truediv__(self, other):
        other = _to_tensor(other)
        bwd = lambda g: (g / other, -g * self / other**2)
        return Tensor(
            ("truediv", self, other, bwd),
            requires_grad=self.requires_grad or other.requires_grad,
            ndim=max(self.ndim, other.ndim),
        )

    def __rtruediv__(self, other):
        return _to_tensor(other) / self

    def __matmul__(self, other):
        other = _to_tensor(other)

        def matmul_bwd(g):
            if self.ndim == 1 and other.ndim == 1:
                return g * other, g * self
            g = g.unsqueeze(1) if g.ndim == 1 else g
            a = self.unsqueeze(1) if self.ndim == 1 else self
            b = other.unsqueeze(1) if other.ndim == 1 else other
            da = g @ b.swapaxes(-1, -2)
            db = a.swapaxes(-1, -2) @ g
            if self.ndim == 1:
                da = da.squeeze()
            if other.ndim == 1:
                db = db.squeeze()
            return da, db

        if self.ndim == 1 and other.ndim == 1:
            out_ndim = 0
        elif self.ndim == 1 or other.ndim == 1:
            out_ndim = max(self.ndim, other.ndim) - 1
        else:
            out_ndim = max(self.ndim, other.ndim)

        return Tensor(
            ("matmul", self, other, matmul_bwd), requires_grad=self.requires_grad or other.requires_grad, ndim=out_ndim
        )

    def __rmatmul__(self, other):
        return _to_tensor(other) @ self

    def __pow__(self, other):
        other = _to_tensor(other)
        bwd = lambda g: (g * other * self ** (other - 1),)
        return Tensor(
            ("pow", self, other, bwd),
            requires_grad=self.requires_grad or other.requires_grad,
            ndim=max(self.ndim, other.ndim),
        )

    def __iadd__(self, other):
        if self.requires_grad:
            raise RuntimeError("in-place operations on tensors with requires_grad=True are not supported")
        self.data = ("add", self.data, _to_tensor(other).data)
        return self

    def __isub__(self, other):
        if self.requires_grad:
            raise RuntimeError("in-place operations on tensors with requires_grad=True are not supported")
        self.data = ("sub", self.data, _to_tensor(other).data)
        return self

    def __eq__(self, other):
        other = _to_tensor(other)
        return Tensor(("eq", self, other), requires_grad=False)

    def __ne__(self, other):
        other = _to_tensor(other)
        return Tensor(("ne", self, other), requires_grad=False)

    def __lt__(self, other):
        other = _to_tensor(other)
        return Tensor(("lt", self, other), requires_grad=False)

    def __gt__(self, other):
        other = _to_tensor(other)
        return Tensor(("gt", self, other), requires_grad=False)

    def __le__(self, other):
        other = _to_tensor(other)
        return Tensor(("le", self, other), requires_grad=False)

    def __ge__(self, other):
        other = _to_tensor(other)
        return Tensor(("ge", self, other), requires_grad=False)

    def __invert__(self):
        return Tensor(("invert", self), requires_grad=False)

    def sum(self, axis=None, keepdims=False):
        return Tensor(("sum", self, axis, keepdims), requires_grad=self.requires_grad, ndim=_reduce_ndim(self.ndim, axis, keepdims))

    def max(self, axis=None, keepdims=False):
        return Tensor(("max", self, axis, keepdims), requires_grad=self.requires_grad, ndim=_reduce_ndim(self.ndim, axis, keepdims))

    def min(self, axis=None, keepdims=False):
        return Tensor(("min", self, axis, keepdims), requires_grad=self.requires_grad, ndim=_reduce_ndim(self.ndim, axis, keepdims))

    def mean(self, axis=None, keepdims=False):
        return Tensor(("mean", self, axis, keepdims), requires_grad=self.requires_grad, ndim=_reduce_ndim(self.ndim, axis, keepdims))

    def transpose(self, axes=None):
        return Tensor(("transpose", self, axes), requires_grad=self.requires_grad, ndim=self.ndim)

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        return Tensor(("swapaxes", self, axis1, axis2), requires_grad=self.requires_grad, ndim=self.ndim)

    def reshape(self, *shape):
        return Tensor(("reshape", self, shape), requires_grad=self.requires_grad, ndim=len(shape))

    def squeeze(self, axis):
        if axis is None:
            raise ValueError("squeeze requires an explicit axis")
        return Tensor(("squeeze", self, axis), requires_grad=self.requires_grad, ndim=self.ndim - 1)

    def unsqueeze(self, axis):
        return Tensor(("unsqueeze", self, axis), requires_grad=self.requires_grad, ndim=self.ndim + 1)

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            out_ndim = self.ndim - 1 + idx.ndim
        elif isinstance(idx, tuple):
            dropped = sum(1 for i in idx if isinstance(i, int))
            out_ndim = self.ndim - dropped
        elif isinstance(idx, int):
            out_ndim = self.ndim - 1
        else:
            out_ndim = self.ndim
        return Tensor(("getitem", self, idx), requires_grad=self.requires_grad, ndim=out_ndim)

    def scatter(self, idx, source):
        source = _to_tensor(source)
        return Tensor(("scatter", self, idx, source), requires_grad=self.requires_grad or source.requires_grad, ndim=self.ndim)


def exp(t):
    t = _to_tensor(t)
    return Tensor(("exp", t), requires_grad=t.requires_grad, ndim=t.ndim)


def log(t):
    t = _to_tensor(t)
    return Tensor(("log", t), requires_grad=t.requires_grad, ndim=t.ndim)


def sqrt(t):
    t = _to_tensor(t)
    return Tensor(("sqrt", t), requires_grad=t.requires_grad, ndim=t.ndim)


def abs(t):
    t = _to_tensor(t)
    return Tensor(("abs", t), requires_grad=t.requires_grad, ndim=t.ndim)


def cat(ts, axis=0):
    return Tensor(("cat", tuple(ts), axis), requires_grad=any(t.requires_grad for t in ts), ndim=ts[0].ndim)


def where(cond, x, y):
    cond = _to_tensor(cond)
    x = _to_tensor(x)
    y = _to_tensor(y)
    return Tensor(
        ("where", cond, x, y), requires_grad=x.requires_grad or y.requires_grad, ndim=max(cond.ndim, x.ndim, y.ndim)
    )


def _reduce_ndim(ndim, axis, keepdims):
    if axis is None:
        return 0
    if keepdims:
        return ndim
    return ndim - 1


def _to_tensor(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False, ndim=0)
