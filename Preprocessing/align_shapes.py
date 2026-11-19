
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to binary femur masks
mask_files = [
    "femur_masks/femur_1_mask.png",
    "femur_masks/femur_2_mask.png",
    "femur_masks/femur_3_mask.png",
    "femur_masks/femur_4_mask.png",
    "femur_masks/femur_5_mask.png",
    "femur_masks/femur_6_mask.png"
]

# Output directory
output_dir = "aligned_femurs"
os.makedirs(output_dir, exist_ok=True)

# Number of contour points
n_points = 100


def extract_contour(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(mask_path)

    # Find external contour
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise ValueError(f"No contour found in {mask_path}")

    # Select largest contour
    contour = max(contours, key=cv2.contourArea)

    # Convert to Nx2 coordinates
    points = contour[:, 0, :].astype(np.float64)

    return points


def resample_contour(points, n_points):
    # Close contour
    points = np.vstack([points, points[0]])

    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))

    cumulative = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative[-1]

    # Equally spaced positions along contour
    samples = np.linspace(0, total_length, n_points, endpoint=False)

    x = np.interp(samples, cumulative, points[:, 0])
    y = np.interp(samples, cumulative, points[:, 1])

    return np.column_stack([x, y])


def center_and_scale(points):
    centered = points - np.mean(points, axis=0)

    scale = np.sqrt(np.sum(centered**2))

    if scale == 0:
        raise ValueError("Invalid contour: zero size")

    return centered / scale



def procrustes_align(reference, target):

    # Center and scale both shapes
    reference = center_and_scale(reference)
    target = center_and_scale(target)

    best_error = float("inf")
    best_aligned = None
    best_shift = None
    best_reversed = None

    # Test both contour directions
    for reverse in [False, True]:

        candidate = target.copy()

        if reverse:
            candidate = candidate[::-1]

        # Test every possible starting point
        for shift in range(len(candidate)):

            shifted = np.roll(candidate, shift, axis=0)

            # Calculate optimal rotation
            H = shifted.T @ reference
            U, S, Vt = np.linalg.svd(H)

            R = U @ Vt

            # Prevent reflection
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            aligned = shifted @ R

            # Calculate alignment error
            error = np.sum((reference - aligned) ** 2)

            if error < best_error:
                best_error = error
                best_aligned = aligned.copy()
                best_shift = shift
                best_reversed = reverse

    print("Best alignment error:", best_error)
    print("Best cyclic shift:", best_shift)
    print("Contour reversed:", best_reversed)

    return reference, best_aligned


# Extract and resample both contours
shapes = []

for mask_path in mask_files:
    contour = extract_contour(mask_path)
    shape = resample_contour(contour, n_points)
    shapes.append(shape)

# Align second shape to first
aligned = []

for i, shape in enumerate(shapes):
    if i == 0:
        aligned.append(center_and_scale(shape))
        continue

    ref, aligned_t = procrustes_align(shapes[0], shapes[i])
    aligned.append(aligned_t)

# Save aligned coordinates
for i, shape in enumerate(aligned, start=1):
    np.save(
        os.path.join(output_dir, f"femur_{i}_aligned.npy"),
        shape
    )

# Visualize all aligned femur shapes
plt.figure(figsize=(8, 8))

for i, shape in enumerate(aligned, start=1):
    plt.plot(
        shape[:, 0], shape[:, 1],
        "-", label=f"Femur {i}"
    )

    plt.scatter(
        shape[:, 0], shape[:, 1],
        s=10
    )

plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()
plt.legend()
plt.title("Procrustes Alignment of Femur Shapes")
plt.show()

print("Alignment completed!")