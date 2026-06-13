import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1
m = 1

# Spatial grid
N = 200
L = 1
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Infinite potential well
V = np.zeros(N)

# Hamiltonian matrix
H = np.zeros((N, N))

for i in range(1, N - 1):
    H[i, i] = 2
    H[i, i - 1] = -1
    H[i, i + 1] = -1

H = -(hbar**2 / (2 * m * dx**2)) * H + np.diag(V)

# Solve eigenvalue problem
E, psi = np.linalg.eigh(H)

# Normalize wavefunctions
for i in range(3):
    psi[:, i] = psi[:, i] / np.sqrt(np.trapz(psi[:, i]**2, x))

# Plot wavefunctions
plt.figure(figsize=(8,5))

for n in range(3):
    plt.plot(x, psi[:, n] + E[n], label=f'n={n}')

plt.title("1D Schrödinger Equation Solver")
plt.xlabel("Position x")
plt.ylabel("Energy")
plt.legend()
plt.grid()

plt.show()
