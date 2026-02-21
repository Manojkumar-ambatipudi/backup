import math

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


def is_primitive_root(a, p):
    if math.gcd(a, p) != 1:
        return False
    
    phi = p - 1
    factors = prime_factors(phi)

    for q in factors:
        if pow(a, phi // q, p) == 1:
            return False
    
    return True


p = 2**61 - 1 

for i in range(1, 75896148945, 2):
    if is_primitive_root(i, p) == True:
        print(i)
