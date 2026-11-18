import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(
    "/Users/lukasweber/Desktop/Artrithis_ML/source/images/Valid/Valid/Osteoarthritis/9000099L.png",
    cv2.IMREAD_GRAYSCALE
)

# Experiment with different settings for the CLAHE algorithm

# settings = [
#     (1.0, (8, 8)),
#     (2.0, (8, 8)),
#     (4.0, (8, 8)),
#     (2.0, (16, 16)),
#     (2.0, (4, 4))
# ]

plt.figure(figsize=(15, 9))

# Visualise different settings -> visually inspection resulted in the usage of clip 2.0 and grid size of 8x8

# for i, (clip, grid) in enumerate(settings):
#
#     clahe = cv2.createCLAHE(
#         clipLimit=clip,
#         tileGridSize=grid
#     )
#
#     result = clahe.apply(image)
#
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(result, cmap="gray")
#     plt.title(
#         f"Clip={clip}, Grid={grid}"
#     )
#     plt.axis("off")
#
# plt.tight_layout()
# plt.show()

# Step 1: CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(image)



