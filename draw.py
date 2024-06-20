import numpy as np
import matplotlib.pyplot as plt

attractor_interval = 6
alpha = 0.7


x = np.linspace(0, 40, 2000)
attractor = (np.floor(x / attractor_interval) + 0.5) * attractor_interval

target_lnr = attractor - alpha * (x - attractor)
target_sin = x - alpha * attractor_interval * np.sin( ( 1 - 2 / attractor_interval * (x - attractor)) * np.pi )





plt.plot(x, target_sin, color='blue')
plt.title('output loss')
plt.show()