import os
import cv2
import numpy as np

# Paths to your two landmark files
landmark_files = [
    "9099961L_femur_landmarks.npy",
    "9115742R_femur_landmarks.npy",
    "9244001R_femur_landmarks.npy",
    "9355390R_femur_landmarks.npy",
    "9422008R_femur_landmarks.npy",
    "9438337R_femur_landmarks.npy",
]


base_str = "/Users/lukasweber/Desktop/Artrithis_ML/source/Preprocessing/annotations/"

output_dir = "femur_masks"
os.makedirs(output_dir, exist_ok=True)

for i, landmark_file in enumerate(landmark_files, start=1):

    landmark = base_str + landmark_file
    # Load landmark coordinates
    points = np.load(landmark).astype(np.int32)

    # Create black mask
    mask = np.zeros([224, 224], dtype=np.uint8)

    # Convert points to OpenCV contour format
    contour = points.reshape((-1, 1, 2))

    # Fill the femur region white
    cv2.fillPoly(mask, [contour], 255)

    # Save mask
    output_path = os.path.join(
        output_dir, f"femur_{i}_mask.png"
    )
    cv2.imwrite(output_path, mask)

    print(f"Saved: {output_path}")

print("Finished creating masks.")