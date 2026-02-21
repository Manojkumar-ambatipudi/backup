import numpy as np
import matplotlib.pyplot as plt

def prng(p, a, seed, n):
    nums = [1] * n
    nums[0] = seed

    for i in range(n - 1):
        nums[i + 1] = (nums[i]*a)%p

    nums = np.array(nums)
    x = nums/p
    return x

def gaussian(x):
    return 1/(np.sqrt(2*np.pi)) * np.exp(-x**2 * 0.5)

p = 2**61 - 1 # chosen a prime number
a = 5593 # chosen a primitive root
print('Give a seed as input: ')
seed = int(input()) 
#seed = 76
n = int(1e7 * 1)

x = prng(p, a, seed, n)
mean , variance = np.mean(x), np.var(x)
print(f'Generated uniform distribution mean {mean: 0.3f} and variance {variance: 0.3f}')

h_array = np.arange(0, 1, 1e-3)
y = np.ones(len(h_array))

plt.figure()
plt.hist(x, bins=300, density=True, color='green', alpha = 0.8, label='Emperical Histogram')
plt.plot(h_array, y, color='blue', alpha = 0.8, label='Theoretical PDF')
plt.legend()
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Uniform Distribution Comparison')
plt.grid(True)
plt.savefig('unfi.png')

x_1, x_2 = x[::2], x[1::2]

x_gauss = np.sqrt(-2*np.log(x_1))*np.sin(2*np.pi*x_2)
mean , variance = np.mean(x_gauss), np.var(x_gauss)
print(f'Generated Gaussian distribution mean {mean: 0.3f} and variance {variance: 0.3f}')
prob = np.mean(np.abs(x_gauss) > 3)
print("Empirical P(|Z|>3):", prob)

h_array = np.arange(-4, 4, 1e-3)
plt.figure()
# Theoretical Gaussian curve
plt.plot(h_array, gaussian(h_array), color='orange', linewidth=2, label='Theoretical PDF')
# Histogram of samples
plt.hist(x_gauss, bins=300, density=True, color='blue', alpha=0.8, label='Empirical Histogram')
plt.legend()
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Gaussian Distribution Comparison')
plt.grid(True)
plt.savefig('gaussian.png')
