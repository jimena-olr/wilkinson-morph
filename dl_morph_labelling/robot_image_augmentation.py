from pathlib import Path
import traceback
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
from src.utils import args

class RobotImageAugmentations:

    def __init__(self):
        self.img = None
        self.rotated_images = None
        self.flipped_images = None
        self.cropped_images = None
        self.augmented_images = None

    def get_image(self, img_path):
        p = Path(img_path)
        if p.exists():
            self.img = cv2.imread(str(p))
        else:
            self.img = None
        return self.img

    def rotating(self, num_rotations):
        if self.img is None:
            return []
        (h, w) = self.img.shape[:2]
        center = (w / 2, h / 2)
        self.rotated_images = []
        rotations = np.linspace(0, 180, num_rotations, endpoint=False, dtype=int).tolist()
        for angle in rotations:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(self.img, M, (w, h))
            self.rotated_images.append(rotated)
        return self.rotated_images

    def flipping(self, img):
        if img is None:
            return []
        flipVertical = cv2.flip(img, 0)
        flipHorizontal = cv2.flip(img, 1)
        flipBoth = cv2.flip(img, -1)
        return [flipVertical, flipHorizontal, flipBoth]

    def do_image_augmentations(self, model_df):
        print('AUGMENTING ROBOT IMAGES')
        save_path = Path('./dl_morph_labelling/images/aug_images/')
        save_path.mkdir(parents=True, exist_ok=True)

        model_df = model_df[model_df['is_valid'] == 0]
        items = [(str(x), int(y)) for x, y in zip(model_df['fname'], model_df['label'])]

        def worker(item):
            try:
                count = 0
                image_path, label = item
                stem = Path(image_path).stem
                img = self.get_image(image_path)
                if img is None:
                    print(f"warning: can't read image {image_path}")
                    return
                rot_images = self.rotating(6)
                for im in rot_images:
                    flips = self.flipping(im)
                    for flipped in flips:
                        file_name = save_path / f'aug{count}_{stem}_{label}.png'
                        cv2.imwrite(str(file_name), flipped)
                        count += 1
            except Exception:
                print(f"error processing {item}:")
                traceback.print_exc()

        Parallel(n_jobs=os.cpu_count())(delayed(worker)(i) for i in tqdm(items, ncols=80))