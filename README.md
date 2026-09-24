# NumPy MLP From Scratch

This is a small deep learning model implemented without PyTorch, TensorFlow,
autograd, or external ML frameworks. The code manually implements:

- Forward propagation through dense and ReLU layers
- Stable softmax cross-entropy loss
- Backward propagation for every layer
- Parameter gradients for weights and biases
- Stochastic gradient descent with momentum and L2 weight decay
- Finite-difference gradient checking
- A runnable training example on a nonlinear spiral dataset

## Run

From the project root:

```powershell
python -B outputs\numpy_mlp_from_scratch.py
```

## Core Math

Dense layer:

```text
Y = XW + b
dW = X.T @ dY
db = sum(dY)
dX = dY @ W.T
```

ReLU:

```text
Y = max(0, X)
dX = dY where X > 0, else 0
```

Softmax cross-entropy:

```text
probs = softmax(logits)
loss = mean(-log(probs[true_class]))
dlogits = (probs - one_hot_labels) / batch_size
```

SGD:

```text
velocity = momentum * velocity - learning_rate * gradient
parameter = parameter + velocity
```

Without momentum:

```text
parameter = parameter - learning_rate * gradient
```

