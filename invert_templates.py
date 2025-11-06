import cv2
import os

template_dir = "templates"

for filename in os.listdir(template_dir):
    if filename.lower().endswith(('.png', '.jpg', '.bmp')):
        path = os.path.join(template_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {filename}")
            continue
        # Invert colors
        img_inv = cv2.bitwise_not(img)
        cv2.imwrite(path, img_inv)
        print(f"Inverted {filename}")

print("All template images have been inverted.")