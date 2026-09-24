*GENERALIZED OUTPUTS
- Outputs are virtually different every single time
- The code spits out something along these lines each time:

```bash
gradient check max relative error: 2.23e-09
epoch    1 | train loss 1.1380, acc 0.514 | val loss 1.0453, acc 0.611
epoch  100 | train loss 0.0157, acc 0.993 | val loss 0.0313, acc 0.986
epoch  200 | train loss 0.0185, acc 0.990 | val loss 0.0226, acc 0.986
epoch  300 | train loss 0.0214, acc 0.990 | val loss 0.0599, acc 0.986
epoch  400 | train loss 0.0166, acc 0.993 | val loss 0.0441, acc 0.986
epoch  500 | train loss 0.0305, acc 0.983 | val loss 0.0509, acc 0.986
epoch  600 | train loss 0.0150, acc 0.997 | val loss 0.0259, acc 0.986
epoch  700 | train loss 0.0127, acc 0.997 | val loss 0.0328, acc 0.986
epoch  800 | train loss 0.0111, acc 0.997 | val loss 0.0339, acc 0.986
epoch  900 | train loss 0.0257, acc 0.986 | val loss 0.0325, acc 0.986
epoch 1000 | train loss 0.0168, acc 0.993 | val loss 0.0266, acc 0.986

first 5 validation predictions:
predicted classes: [1 1 2 1 0]
true classes:      [1 1 2 1 0]
class probabilities:
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]

```
