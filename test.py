import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create meshgrid
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Loss landscape
Z = 1.5 * np.sin(2 * np.pi * X) * np.exp(-((Y - 0.8)**2) * 5) + (Y - 0.5)**2

# Create figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot surface (semi-transparent)
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.95)

# Define optimal point
opt_x, opt_y = 0.85, 0.75
opt_z = 1.5 * np.sin(2 * np.pi * opt_x) * np.exp(-((opt_y - 0.8)**2) * 5) + (opt_y - 0.5)**2

# Generate paths
t = np.linspace(0, 1, 40)

# Stable (smooth) path
path_x_stable = 0.1 + 0.75 * t
path_y_stable = 0.2 + 0.55 * (t ** 1.5)
path_z_stable = 1.5 * np.sin(2 * np.pi * path_x_stable) * np.exp(-((path_y_stable - 0.8)**2) * 5) + (path_y_stable - 0.5)**2

# mild
path_x_osc = 0.1 + 0.75 * t + 0.03 * np.sin(5 * np.pi * t)  # smaller amplitude, lower freq
path_y_osc = 0.2 + 0.55 * (t ** 1.3)
path_z_osc = 1.5 * np.sin(2 * np.pi * path_x_osc) * np.exp(-((path_y_osc - 0.8)**2) * 5) + (path_y_osc - 0.5)**2


# strong
path_x_osc_strong = 0.1 + 0.75 * t + 0.08 * np.sin(10 * np.pi * t)
path_x_osc_strong = 0.2 + 0.55 * (t ** 1.2) + 0.02 * np.sin(6 * np.pi * t)
path_z_osc_strong = 1.5 * np.sin(2 * np.pi * path_x_osc) * np.exp(-((path_y_osc - 0.8)**2) * 5) + (path_y_osc - 0.5)**2

# very strong
# Strongly oscillating (unstable) path — big amplitude + high frequency
path_x_osc_vstrong = 0.1 + 0.75 * t + 0.12 * np.sin(16 * np.pi * t)
path_y_osc_vstrong = 0.2 + 0.55 * (t ** 1.1) + 0.04 * np.sin(10 * np.pi * t)
path_z_osc_vstrong = 1.5 * np.sin(2 * np.pi * path_x_osc) * np.exp(-((path_y_osc - 0.8)**2) * 5) + (path_y_osc - 0.5)**2


# Plot both paths
# ax.plot(path_x_stable, path_y_stable, path_z_stable + 0.02, color='blue', lw=3, label='Stable Path')
# ax.plot(path_x_osc, path_y_osc, path_z_osc + 0.02, color='black', lw=2.5, label='Oscillating Path')
ax.plot(path_x_osc_strong, path_x_osc_strong, path_z_osc_strong + 0.02, color='red', lw=2.5, label='Oscillating Path')
ax.plot(path_x_osc_vstrong, path_y_osc_vstrong, path_z_osc_vstrong + 0.02, color='black', lw=2.5, label='Strongly Oscillating Path')


path_x_osc_extreme = 0.1 + 0.75 * t + 0.18 * np.sin(24 * np.pi * t)  # higher amplitude + frequency
path_y_osc_extreme = 0.2 + 0.55 * (t ** 1.05) + 0.07 * np.sin(14 * np.pi * t)
path_z_osc_extreme = 1.5 * np.sin(2 * np.pi * path_x_osc_extreme) * np.exp(-((path_y_osc_extreme - 0.8)**2) * 5) + (path_y_osc_extreme - 0.5)**2

ax.plot(path_x_osc_extreme, path_y_osc_extreme, path_z_osc_extreme + 0.03, 
        color='purple', lw=2.8, label='Extremely Oscillating Path')

# Optimal point
ax.scatter(opt_x, opt_y, opt_z, color='red', s=60, label='Optimal Hyperparameter')

# Labels and styling
ax.set_xlabel('Human Evaluation Score')
ax.set_ylabel('p')
ax.set_zlabel('Automatic Evaluation Score')
ax.set_title('Loss Landscape Convergence: Stable vs. Oscillating Paths')

# Colorbar and legend
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
ax.legend()

plt.tight_layout()
plt.savefig("image_stable_vs_oscillating.png")
