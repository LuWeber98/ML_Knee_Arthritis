

import os
import cv2
import numpy as np
import matplotlib

# Use a separate interactive window in PyCharm
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


# =====================================================
# 1. SETTINGS
# =====================================================

image_path = "/Users/lukasweber/Desktop/Artrithis_ML/source/images/test/test/Normal/9000622L.png"

bone_name = "femur"
n_landmarks = 20

# Save annotations in a folder next to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "annotations")

os.makedirs(output_dir, exist_ok=True)


# =====================================================
# 2. LOAD IMAGE
# =====================================================

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

print("Image loaded successfully!")
print("Image dimensions:", image.shape)


# =====================================================
# 3. APPLY CLAHE
# =====================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

image = clahe.apply(image)


# =====================================================
# 4. INITIALIZE ANNOTATION
# =====================================================

landmarks = []

fig, ax = plt.subplots(figsize=(10, 8))


# =====================================================
# 5. UPDATE PLOT
# =====================================================

def update_plot():

    ax.clear()

    ax.imshow(image, cmap="gray")
    ax.axis("off")

    if len(landmarks) > 0:

        points = np.array(landmarks)

        # Draw red landmark points
        ax.plot(
            points[:, 0],
            points[:, 1],
            "ro",
            markersize=7
        )

        # Draw blue contour connecting landmarks
        if len(landmarks) >= 2:

            contour = points.copy()

            # Close contour after all 20 points are placed
            if len(landmarks) == n_landmarks:
                contour = np.vstack(
                    [contour, contour[0]]
                )

            ax.plot(
                contour[:, 0],
                contour[:, 1],
                "b-",
                linewidth=2
            )

        # Add landmark numbers
        for i, (x, y) in enumerate(landmarks):

            ax.text(
                x + 5,
                y,
                str(i + 1),
                color="yellow",
                fontsize=10
            )

    ax.set_title(
        f"{bone_name.capitalize()} Annotation | "
        f"Landmarks: {len(landmarks)}/{n_landmarks}\n"
        "Left click: Add | U: Undo | R: Restart | S: Save"
    )

    fig.canvas.draw_idle()


# =====================================================
# 6. ADD LANDMARKS WITH MOUSE
# =====================================================

def onclick(event):

    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    if len(landmarks) >= n_landmarks:

        print("All 20 landmarks have been placed!")
        return

    x = int(event.xdata)
    y = int(event.ydata)

    landmarks.append([x, y])

    print(
        f"Landmark {len(landmarks)}: "
        f"x={x}, y={y}"
    )

    update_plot()


# =====================================================
# 7. KEYBOARD CONTROLS
# =====================================================

def onkey(event):

    key = event.key.lower() if event.key else ""

    # Undo last landmark
    if key == "u":

        if landmarks:

            removed = landmarks.pop()

            print("Removed landmark:", removed)

        update_plot()

    # Restart annotation
    elif key == "r":

        landmarks.clear()

        print("All landmarks removed.")

        update_plot()

    # Save coordinates
    elif key == "s":

        if len(landmarks) != n_landmarks:

            print(
                f"Please place all {n_landmarks} "
                "landmarks before saving."
            )

            return

        coordinates = np.array(
            landmarks,
            dtype=np.int32
        )

        base_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        filename = (
            f"{base_name}_{bone_name}_landmarks"
        )

        # Save as NumPy file
        np.save(
            os.path.join(
                output_dir,
                filename + ".npy"
            ),
            coordinates
        )

        # Save as CSV file
        np.savetxt(
            os.path.join(
                output_dir,
                filename + ".csv"
            ),
            coordinates,
            delimiter=",",
            header="x,y",
            comments="",
            fmt="%d"
        )

        print("\nLandmarks successfully saved!")
        print(coordinates)

        print(
            "\nFiles saved in:",
            output_dir
        )


# =====================================================
# 8. CONNECT MOUSE AND KEYBOARD EVENTS
# =====================================================

fig.canvas.mpl_connect(
    "button_press_event",
    onclick
)

fig.canvas.mpl_connect(
    "key_press_event",
    onkey
)


# =====================================================
# 9. START ANNOTATION
# =====================================================

update_plot()

plt.show()
