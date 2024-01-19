import matplotlib.pyplot as plt

# Data for the first graph
wind_speeds = [1, 5, 7.5, 10, 12.5, 15]
d_1 = [4.009792188, 57.27047835, 189.3508343, 346.9693513, 526.4182943, 1040.107799]
d_2 = [568.2913308, 600.8888812, 643.085092, 1025.130869, 1445.082062, 2995.947699]

# Data for the second graph
disparity_percent = [21.95, 22.03, 23.16, 23.87, 25.97, 27.95]

# Create the first graph
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(wind_speeds, d_1, label='$d_1$')
plt.scatter(wind_speeds, d_2, label='$d_2$')
plt.xlabel('Wind Speed $(m.s^{-1})$', fontsize = 15)
plt.ylabel('Minimal distance (m)', fontsize = 15)

plt.legend(fontsize = 15)
plt.grid()
# Create the second graph
plt.subplot(1, 2, 2)
plt.scatter(wind_speeds, disparity_percent, color='red')
plt.xlabel('Wind Speed $(m.s^{-1})$', fontsize = 15)
plt.ylabel('Flight Time Relative Error (%)', fontsize = 15)

plt.legend(fontsize = 15)
plt.grid()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
