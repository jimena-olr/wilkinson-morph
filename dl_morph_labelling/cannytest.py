import random, os
import cv2
from preprocessing import canny_vial_crop

noncrystal_dir = "C:\\Penn\\Zahrt Lab Projects\\wilkinson-morph\\wilkinson-morph\\dl_morph_labelling\\C3\\precipitate"

for i in range(5):
    fname = random.choice(os.listdir(noncrystal_dir))
    img_path = os.path.join(noncrystal_dir, fname)

    img = cv2.imread(img_path)
    cropped = canny_vial_crop(img)

    cv2.imwrite(f"debug_orig_{i}.png", img)
    cv2.imwrite(f"debug_crop_{i}.png", cropped)

    print("Processed:", img_path)
