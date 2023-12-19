import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


""" Problem 6.1 """
A = np.array([[0, 2, 4], [2, 4, 2], [3, 3, 1]])
b = np.array([[-2], [-2], [-4]])
c = np.array([[1], [1], [1]])
A_inv = np.linalg.inv(A)
# Part a
print("Part (a)")
print(f"A^-1 = {A_inv}")
print(" ")

# Part b
print("Part (b)")
print(f"A^-1.b = {A_inv @ b}")
print(f"A.c = {A @ c}")


""" Problem 6.2 """
# Part a
n = 40000
z = np.random.randn(n)
sns.set_theme()
fig, ax = plt.subplots(1, 1)
ax.step(sorted(z), np.arange(1, n + 1) / float(n), label="Gaussian")

# Part b
k_arr = [1, 8, 64, 512]
for k in k_arr:
    yk = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1 / k), axis=1)
    ax.step(sorted(yk), np.arange(1, n + 1) / float(n), label=str(k))
# Axes limits, labels, and title
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1)
ax.set_xlabel("Observations")
ax.set_ylabel("Probability")
ax.legend()
plt.savefig("./6.2.pdf", bbox_inches="tight")
plt.show()

