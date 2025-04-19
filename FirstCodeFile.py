import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 100)
y = 1 + (0-1)*np.cos(x)*np.exp(-x/3)

plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()