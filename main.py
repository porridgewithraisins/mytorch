import tensa as t
import mytorch as m

t.np.random.seed(42)

B, D_in, D_h1, D_h2, D_out = 4, 8, 16, 16, 3

x  = t.tensor(t.np.random.randn(B, D_in))
W1 = t.tensor(t.np.random.randn(D_in, D_h1) * 0.1, requires_grad=True)
b1 = t.tensor(t.np.zeros(D_h1), requires_grad=True)
W2 = t.tensor(t.np.random.randn(D_h1, D_h2) * 0.1, requires_grad=True)
b2 = t.tensor(t.np.zeros(D_h2), requires_grad=True)
W3 = t.tensor(t.np.random.randn(D_h2, D_out) * 0.1, requires_grad=True)
b3 = t.tensor(t.np.zeros(D_out), requires_grad=True)

y = t.tensor(t.np.array([0, 1, 2, 0]))

print("W1.requires_grad:", W1.requires_grad)
print("x.requires_grad:", x.requires_grad)
print("_is_grad_disabled():", t._is_grad_disabled())

print("(x @ W1).requires_grad:", (x @ W1).requires_grad)
h1 = m.relu(x @ W1 + b1)
print("h1.requires_grad:", h1.requires_grad)
h2 = m.tanh(h1 @ W2 + b2)
logits = h2 @ W3 + b3

log_probs = m.log_softmax(logits, axis=-1)
loss = -log_probs[t.np.arange(B), y.data].mean()

print("loss:", loss.item())

print("loss.requires_grad:", loss.requires_grad)
print("logits.requires_grad:", logits.requires_grad)
print("log_probs.requires_grad:", log_probs.requires_grad)

loss.backward()

for name, param in [("W1", W1), ("W2", W2), ("W3", W3), ("b1", b1), ("b2", b2), ("b3", b3)]:
    print(f"{name}.grad:", param.grad)

for name, param in [("W1", W1), ("W2", W2), ("W3", W3), ("b1", b1), ("b2", b2), ("b3", b3)]:
    print(f"{name}.grad norm:", t.np.linalg.norm(param.grad.data))