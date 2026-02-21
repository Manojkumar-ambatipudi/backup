import numpy as np
import matplotlib.pyplot as plt
import math

def unif(a, m, num_samples, seed = 42):
    x = [1] * num_samples
    x[0] = seed
    for i in range(num_samples - 1):
        x[i+1] = (a*x[i])%m
    x = np.array(x)
    x = x/m
    return x

def prime_factors(n):
    factors = set()
    while n % 2 == 0:
        factors.add(2)
        n //= 2
    f = 3
    while f * f <= n:
        while n % f == 0:
            factors.add(f)
            n //= f
        f += 2
    if n > 1:
        factors.add(n)  
    return factors

def isprimitiveroot(i, m):
    if math.gcd(i, m) != 1:
        return False
    
    k = m - 1
    s = prime_factors(k)

    for factor in s:
        if pow(i, k // factor, m) == 1:
            return False
    return True

samples = int(2*1e8)
seed = 42
m = 2**61 - 1
primitive_roots = []
for i in range(10**4):
    if isprimitiveroot(i, m) == True:
        primitive_roots.append(i)

a = primitive_roots[-3]
print(a)


U = unif(a, m, samples)
plt.figure()
plt.hist(U, density=True)
plt.grid(True)
plt.savefig('unif.png')

sigma = 1
X_rayleigh = np.sqrt(-2*sigma**2*np.log(1 - U))
plt.figure()
plt.hist(X_rayleigh, density=True, bins=500)
plt.grid(True)
plt.savefig('ray.png')


