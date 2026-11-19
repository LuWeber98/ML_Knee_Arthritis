
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import masked_singleton

input_dir = "aligned_femurs"

shapes = np.array([
    np.load(os.path.join(input_dir, f"femur_{i}_aligned.npy"))
    for i in range(1, 7)
])

print("Shape array dimensions:", shapes.shape)

# Calculate mean shape
mean_shape = np.mean(shapes, axis=0)

print("Mean shape dimensions:", mean_shape.shape)

# Save mean shape
np.save(
    os.path.join(input_dir, "mean_femur_shape.npy"),
    mean_shape
)

# Close contour for visualization
mean_closed = np.vstack([
    mean_shape,
    mean_shape[0]
])

# Visualize
plt.figure(figsize=(8, 8))

plt.plot(
    mean_closed[:, 0],
    mean_closed[:, 1],
    "g-",
    linewidth=2,
    label="Mean Femur Shape"
)

plt.scatter(
    mean_shape[:, 0],
    mean_shape[:, 1],
    color="green",
    s=10
)

plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()

plt.title("Mean Femur Shape")
plt.legend()
plt.show()

print("Mean shape successfully calculated!")
