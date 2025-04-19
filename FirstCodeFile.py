import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 15, 100)
# Generate a sine wave with an exponential decay
v0 = 0
v1 = 1
wn = .2
zeta = 0.05
y = v1 + (v0-v1)*np.cos(wn*x)*np.exp(-zeta*x)

plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()