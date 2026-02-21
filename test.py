# Visualization of the set S for m = 2
# S = {(x1,x2) : |x1 cos(t) + x2 cos(2t)| <= 1 for all |t| <= pi/3}

import numpy as np
import matplotlib.pyplot as plt

# Discretize t
t_vals = np.linspace(-np.pi/3, np.pi/3, 400)

# Create grid in (x1, x2)
x1 = np.linspace(-3, 3, 400)
x2 = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1, x2)

# Check constraint
mask = np.ones_like(X1, dtype=bool)

for t in t_vals:
    p = X1 * np.cos(t) + X2 * np.cos(2*t)
    mask &= (np.abs(p) <= 1)

# Plot region
plt.figure()
plt.contourf(X1, X2, mask.astype(int))
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Region S for m=2")
plt.savefig('test.png')
