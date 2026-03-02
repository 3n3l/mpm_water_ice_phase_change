import numpy as np
from matplotlib import pyplot as plt


def cubic_kernel(r):
    ar = np.abs(r)
    if 0 <= ar <= 1:
        return 0.5 * ar**3 - ar**2 + 0.67
    elif 1 <= ar <= 2:
        return 0.167 * (2 - ar) ** 3
    else:
        return 0


def quadratic_kernel(r):
    ar = np.abs(r)
    if 0 <= ar <= 1 / 2:
        return (3 / 4) - ar**2
    elif 1 / 2 <= ar <= 3 / 2:
        return (1 / 2) * ((3 / 2) - ar) ** 2
    else:
        return 0


def integral_cubic_kernel(r):
    if r < -2.0:
        return 0
    elif r < -1.0:
        return 1 / 24 * r**4 + 1 / 3 * r**3 + r**2 + 4 / 3 * r + 2 / 3
    elif r < 0.0:
        return -1 / 8 * r**4 - 1 / 3 * r**3 + 2 / 3 * r + 1 / 2
    elif r < 1.0:
        return 1 / 8 * r**4 - 1 / 3 * r**3 + 2 / 3 * r + 1 / 2
    elif r < 2.0:
        return -1 / 24 * r**4 + 1 / 3 * r**3 - r**2 + 4 / 3 * r + 1 / 3
    else:
        return 1


def integral_quadratic_kernel(r):
    if r < -1.5:
        return 0
    elif r < -0.5:
        return 1 / 6 * (3 / 2 + r) ** 3
    elif r < 0.5:
        return -1 / 3 * r**3 + 3 / 4 * r + 1 / 2
    elif r < 1.5:
        return -1 / 6 * (3 / 2 - r) ** 3 + 1
    else:
        return 1


x_cubic = np.linspace(-2, 2, 100)
x_quadratic = np.linspace(-1.5, 1.5, 100)

y_cubic_integral = [integral_cubic_kernel(r) for r in x_cubic]
y_quadratic_integral = [integral_quadratic_kernel(r) for r in x_quadratic]
y_cubic_func = [cubic_kernel(r) for r in x_cubic]
y_quadratic_func = [quadratic_kernel(r) for r in x_quadratic]

plt.figure(figsize=(12, 5))

# plt.subplot(2, 2, 1)
plt.plot(x_cubic, y_cubic_func, label="Cubic Kernel", color="purple")
# plt.title("Cubic Kernel")
# plt.xlabel("r")
# plt.ylabel("Value")
# plt.grid()
# plt.legend()

# plt.subplot(2, 2, 2)
plt.plot(x_quadratic, y_quadratic_func, label="Quadratic Kernel", color="red")
# plt.title("Quadratic Kernel")
# plt.xlabel("r")
# plt.ylabel("Value")
# plt.grid()
# plt.legend()

# plt.subplot(2, 2, 3)
plt.plot(x_cubic, y_cubic_integral, label="Integral Cubic Kernel", color="orange")
# plt.title("Integral of Cubic Kernel")
# plt.xlabel("r")
# plt.ylabel("Integral Value")
# plt.grid()
# plt.legend()

# plt.subplot(2, 2, 4)
plt.plot(x_quadratic, y_quadratic_integral, label="Integral Quadratic Kernel", color="blue")
# plt.title("Integral of Quadratic Kernel")
# plt.xlabel("r")
# plt.ylabel("Integral Value")
# plt.grid()
# plt.legend()

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
