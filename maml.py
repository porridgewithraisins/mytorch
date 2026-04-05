import tensa as t
import mytorch as m

t.np.random.seed(42)

# outer params
w = t.tensor(t.np.random.randn(4, 2) * 0.1, requires_grad=True)
print(f"w.requires_grad: {w.requires_grad}")
print(f"w.shape: {w.shape}")


def inner_loss(w, x, y):
    pred = x @ w
    loss = ((pred - y) ** 2).mean()
    return loss


def outer_loss(w, tasks):
    total = t.tensor(0.0)
    for i, (x, y, x_val, y_val) in enumerate(tasks):
        # inner loop
        loss = inner_loss(w, x, y)
        print(f"\n[Task {i}] inner loss: {loss.item():.4f}")
        print(f"[Task {i}] loss.requires_grad: {loss.requires_grad}")

        loss.backward(higher_order=True)

        print(f"[Task {i}] w.grad is None: {w.grad is None}")
        print(f"[Task {i}] w.grad.requires_grad: {w.grad.requires_grad}")
        print(f"[Task {i}] w.grad has parents: {not _is_leaf(w.grad)}")

        w_adapted = w - 0.01 * w.grad
        print(f"[Task {i}] w_adapted.requires_grad: {w_adapted.requires_grad}")
        print(f"[Task {i}] w_adapted has parents: {not _is_leaf(w_adapted)}")

        val_loss = inner_loss(w_adapted, x_val, y_val)
        print(f"[Task {i}] val_loss: {val_loss.item():.4f}")
        print(f"[Task {i}] val_loss.requires_grad: {val_loss.requires_grad}")

        total = total + val_loss

        # reset grad for next task
        w.grad = None

    return total / len(tasks)


# need _is_leaf for debug
from tensa import _is_leaf

# fake tasks
tasks = [
    (
        t.tensor(t.np.random.randn(3, 4)),
        t.tensor(t.np.random.randn(3, 2)),
        t.tensor(t.np.random.randn(3, 4)),
        t.tensor(t.np.random.randn(3, 2)),
    )
    for _ in range(4)
]

print("\n=== Running outer loop ===")
meta_loss = outer_loss(w, tasks)
print(f"\nmeta_loss: {meta_loss.item():.4f}")
print(f"meta_loss.requires_grad: {meta_loss.requires_grad}")

print("\n=== Running meta backward ===")
meta_loss.backward()

print(f"\nw.grad is None: {w.grad is None}")
print(f"w.grad.requires_grad: {w.grad.requires_grad}")
print(f"w.grad:\n{w.grad}")
print(f"w.grad norm: {t.np.linalg.norm(w.grad.data):.6f}")
