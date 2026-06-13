import numpy as np
import matplotlib.pyplot as plt

# Spatial grid
N = 300
x = np.linspace(-5, 5, N)
dx = x[1] - x[0]

# Harmonic oscillator potential
V = 0.5 * x**2

# Hamiltonian matrix
H = np.zeros((N, N))

for i in range(1, N - 1):
    H[i, i] = 2 + dx**2 * V[i]
    H[i, i - 1] = -1
    H[i, i + 1] = -1

H = H / dx**2

# Solve eigenvalue problem
E, psi = np.linalg.eigh(H)

# Plot first 4 wavefunctions
plt.figure(figsize=(8,5))

for n in range(4):
    plt.plot(x, psi[:, n] + E[n], label=f'n={n}')

plt.title("Quantum Harmonic Oscillator")
plt.xlabel("Position x")
plt.ylabel("Energy")
plt.legend()
plt.grid()

plt.show()
