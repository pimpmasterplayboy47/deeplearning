"""
NumPy-only deep learning model with manual forward propagation,
backward propagation, and SGD optimization.


No PyTorch, TensorFlow, autograd, or external ML libraries are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

Array = np.ndarray

class Dense:
    """Fully connected layer: out = x @ W + b."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        # He initialization works well with ReLU hidden layers.
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, scale, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))

        self.x: Array | None = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: Array) -> Array:
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out: Array) -> Array:
        if self.x is None:
            raise RuntimeError("forward must be called before backward")

        # For y = x @ W + b:
        # dL/dW = x.T @ dL/dy
        # dL/db = sum(dL/dy)
        # dL/dx = dL/dy @ W.T
        self.dW = self.x.T @ grad_out
        self.db = np.sum(grad_out, axis=0, keepdims=True)
        return grad_out @ self.W.T

    def parameters(self, prefix: str) -> Iterable[tuple[str, Array, Array]]:
        yield f"{prefix}.W", self.W, self.dW
        yield f"{prefix}.b", self.b, self.db

class ReLU:
    """Rectified linear unit activation: max(0, x)."""

    def __init__(self):
        self.mask: Array | None = None

    def forward(self, x: Array) -> Array:
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out: Array) -> Array:
        if self.mask is None:
            raise RuntimeError("forward must be called before backward")
        return grad_out * self.mask

    def parameters(self, prefix: str) -> Iterable[tuple[str, Array, Array]]:
        return ()


class SoftmaxCrossEntropy:
    """Softmax activation plus cross-entropy loss for integer labels."""

    def __init__(self):
        self.probs: Array | None = None
        self.y: Array | None = None

    def forward(self, logits: Array, y: Array) -> float:
        # Stable softmax: subtract max logit before exponentiating.
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.y = y

        n = logits.shape[0]
        correct_log_probs = -np.log(self.probs[np.arange(n), y] + 1e-12)
        return float(np.mean(correct_log_probs))

    def backward(self) -> Array:
        if self.probs is None or self.y is None:
            raise RuntimeError("forward must be called before backward")

        # If L = mean(-log(softmax(logits)[true_class])),
        # then dL/dlogits = (softmax_probs - one_hot_labels) / batch_size.
        n = self.probs.shape[0]
        grad_logits = self.probs.copy()
        grad_logits[np.arange(n), self.y] -= 1.0
        grad_logits /= n
        return grad_logits


class MLP:
    """A feed-forward neural network made from Dense and ReLU layers."""

    def __init__(self, layer_sizes: list[int], seed: int = 0):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain input and output sizes")

        rng = np.random.default_rng(seed)
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], rng))
            if i < len(layer_sizes) - 2:
                self.layers.append(ReLU())

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: Array) -> Array:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> Iterable[tuple[str, Array, Array]]:
        dense_index = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                yield from layer.parameters(f"dense_{dense_index}")
                dense_index += 1

    def predict(self, x: Array) -> Array:
        logits = self.forward(x)
        return np.argmax(logits, axis=1)


@dataclass
class SGD:
    """Stochastic gradient descent with optional momentum and weight decay."""

    learning_rate: float = 0.1
    momentum: float = 0.0
    weight_decay: float = 0.0

    def __post_init__(self):
        self.velocity: dict[str, Array] = {}

    def step(self, parameters: Iterable[tuple[str, Array, Array]]) -> None:
        for name, param, grad in parameters:
            update_grad = grad

            # L2 regularization: add lambda * W to the weight gradient.
            # Bias vectors are left unregularized.
            if self.weight_decay and param.ndim > 1:
                update_grad = update_grad + self.weight_decay * param

            if self.momentum:
                if name not in self.velocity:
                    self.velocity[name] = np.zeros_like(param)
                self.velocity[name] = (
                    self.momentum * self.velocity[name]
                    - self.learning_rate * update_grad
                )
                param += self.velocity[name]
            else:
                param -= self.learning_rate * update_grad


def make_spiral_data(
    samples_per_class: int = 120,
    classes: int = 3,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[Array, Array]:
    """Create a small nonlinear classification dataset."""

    rng = np.random.default_rng(seed)
    total = samples_per_class * classes
    x = np.zeros((total, 2))
    y = np.zeros(total, dtype=np.int64)

    for class_id in range(classes):
        row_slice = slice(
            class_id * samples_per_class,
            (class_id + 1) * samples_per_class,
        )
        radius = np.linspace(0.0, 1.0, samples_per_class)
        theta = (
            np.linspace(class_id * 4.0, (class_id + 1) * 4.0, samples_per_class)
            + rng.normal(0.0, noise, samples_per_class)
        )
        x[row_slice] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[row_slice] = class_id

    return x, y


def standardize(train_x: Array, val_x: Array) -> tuple[Array, Array]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-8
    return (train_x - mean) / std, (val_x - mean) / std


def accuracy(model: MLP, x: Array, y: Array) -> float:
    return float(np.mean(model.predict(x) == y))


def evaluate(model: MLP, loss_fn: SoftmaxCrossEntropy, x: Array, y: Array) -> tuple[float, float]:
    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)
    return loss, accuracy(model, x, y)


def train(
    model: MLP,
    train_x: Array,
    train_y: Array,
    val_x: Array,
    val_y: Array,
    epochs: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 0.2,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    seed: int = 7,
    log_every: int = 100,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    optimizer = SGD(learning_rate, momentum, weight_decay)
    loss_fn = SoftmaxCrossEntropy()
    history = []

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(train_x))

        for start in range(0, len(train_x), batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = train_x[batch_idx]
            yb = train_y[batch_idx]

            logits = model.forward(xb)
            loss_fn.forward(logits, yb)
            grad_logits = loss_fn.backward()
            model.backward(grad_logits)
            optimizer.step(model.parameters())

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            train_loss, train_acc = evaluate(model, loss_fn, train_x, train_y)
            val_loss, val_acc = evaluate(model, loss_fn, val_x, val_y)
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history.append(row)
            print(
                f"epoch {epoch:4d} | "
                f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.3f}"
            )

    return history


def gradient_check(seed: int = 123, eps: float = 1e-5, checks: int = 20) -> float:
    """Compare manual gradients with finite-difference gradients."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(5, 2))
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    model = MLP([2, 6, 3], seed=seed)
    loss_fn = SoftmaxCrossEntropy()

    logits = model.forward(x)
    loss_fn.forward(logits, y)
    model.backward(loss_fn.backward())

    max_relative_error = 0.0

    for name, param, grad in model.parameters():
        flat_indices = rng.choice(param.size, size=min(checks, param.size), replace=False)

        for flat_index in flat_indices:
            index = np.unravel_index(flat_index, param.shape)
            old_value = param[index]

            param[index] = old_value + eps
            loss_plus = loss_fn.forward(model.forward(x), y)

            param[index] = old_value - eps
            loss_minus = loss_fn.forward(model.forward(x), y)

            param[index] = old_value
            numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
            manual_grad = grad[index]

            denom = max(1e-12, abs(numerical_grad) + abs(manual_grad))
            relative_error = abs(numerical_grad - manual_grad) / denom
            max_relative_error = max(max_relative_error, relative_error)

    return max_relative_error


def main() -> None:
    max_grad_error = gradient_check()
    print(f"gradient check max relative error: {max_grad_error:.2e}")

    x, y = make_spiral_data()
    rng = np.random.default_rng(99)
    split_indices = rng.permutation(len(x))
    train_count = int(0.8 * len(x))
    train_idx = split_indices[:train_count]
    val_idx = split_indices[train_count:]

    train_x, train_y = x[train_idx], y[train_idx]
    val_x, val_y = x[val_idx], y[val_idx]
    train_x, val_x = standardize(train_x, val_x)

    model = MLP([2, 64, 64, 3], seed=1)
    train(
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=1000,
        batch_size=64,
        learning_rate=0.2,
        momentum=0.9,
        weight_decay=1e-4,
        log_every=100,
    )

    sample_probs = SoftmaxCrossEntropy()
    logits = model.forward(val_x[:5])
    sample_probs.forward(logits, val_y[:5])
    print("\nfirst 5 validation predictions:")
    print("predicted classes:", np.argmax(sample_probs.probs, axis=1))
    print("true classes:     ", val_y[:5])
    print("class probabilities:")
    print(np.round(sample_probs.probs, 3))


if __name__ == "__main__":
    main()
