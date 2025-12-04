# --- append to preprocessing.py ---
import pandas as pd
import cv2, numpy as np, random, csv
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _imread_any(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

def _imsave_any(p, img):
    # robust save for windows/unicode paths
    p = Path(p)
    _ensure_dir(p.parent)
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(p))

def _center_rect_crop(img, frac=0.90):
    h, w = img.shape[:2]
    th, tw = int(h*frac), int(w*frac)
    y0 = (h - th)//2; x0 = (w - tw)//2
    return img[y0:y0+th, x0:x0+tw]

import math

def canny_vial_crop(img, inside_frac=0.7, min_size=64):
    """
    Circle-based crop for NON-crystals, with the crop guaranteed
    to lie fully *inside* the detected well circle.

    Steps:
      1) grayscale + blur
      2) HoughCircles to detect the well
      3) shrink the radius to inside_frac
      4) take the largest axis-aligned square that fits inside that circle
      5) fallback to center crop if anything fails
    """
    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    min_side = min(h_img, w_img)
    min_radius = int(0.35 * min_side)
    max_radius = int(0.60 * min_side)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(0.8 * min_side),
        param1=100,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None or len(circles[0]) == 0:
        return _center_rect_crop(img, 0.88)
    x_c, y_c, r = circles[0][0]
    x_c, y_c, r = float(x_c), float(y_c), float(r)

    r_safe = r * float(inside_frac)
    half_side = int(r_safe / math.sqrt(2))

    if half_side < min_size // 2:
        return _center_rect_crop(img, 0.88)

    x0 = int(round(x_c)) - half_side
    y0 = int(round(y_c)) - half_side
    x1 = int(round(x_c)) + half_side
    y1 = int(round(y_c)) + half_side

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w_img, x1)
    y1 = min(h_img, y1)

    crop = img[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < min_size:
        return _center_rect_crop(img, 0.88)

    return crop


def preprocess_crystal_dir(in_dir, out_dir, out_size=(224,224)):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    _ensure_dir(out_dir)
    saved = []
    for p in in_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".bmp"}:
            img = _imread_any(p)
            # light normalization: center crop (or your rim-removal if you have it)
            img = _center_rect_crop(img, 0.92)
            img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
            out_p = out_dir / (p.stem + ".png")
            _imsave_any(out_p, img)
            saved.append(str(out_p))
    return saved

def preprocess_noncrystal_dir_canny(in_dir, out_dir, out_size=(224,224)):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    _ensure_dir(out_dir)
    saved = []
    for p in in_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".bmp"}:
            img = _imread_any(p)
            img = canny_vial_crop(img)
            img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
            out_p = out_dir / (p.stem + ".png")
            _imsave_any(out_p, img)
            saved.append(str(out_p))
    return saved

def build_head1_manifest(crystal_pp_dir, noncrystal_pp_dir, out_csv, balance=True, seed=42):
    rng = random.Random(seed)
    c_paths = sorted(str(p) for p in Path(crystal_pp_dir).glob("*.png"))
    n_paths = sorted(str(p) for p in Path(noncrystal_pp_dir).glob("*.png"))
    if balance:
        k = min(len(c_paths), len(n_paths))
        c_paths = rng.sample(c_paths, k)
        n_paths = rng.sample(n_paths, k)
    rows = [(p, 1) for p in c_paths] + [(p, 0) for p in n_paths]
    rng.shuffle(rows)
    _ensure_dir(Path(out_csv).parent)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname","label"])
        w.writerows(rows)
    return out_csv

def add_stratified_folds(manifest_csv, n_splits=5, seed=42):
    df = pd.read_csv(manifest_csv)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for k, (_, val_idx) in enumerate(skf.split(df["fname"], df["label"])):
        df.loc[val_idx, "fold"] = k
    out_csv = str(Path(manifest_csv).with_name(Path(manifest_csv).stem + f"_folds{n_splits}.csv"))
    df.to_csv(out_csv, index=False)
    return out_csv

def df_from_manifest_fold(fold_csv, fold_k):
    """Return train/val DataFrame with columns fname,label,is_valid for robot_kfold_fastai."""
    df = pd.read_csv(fold_csv)
    train = df[df.fold != fold_k].copy()
    val   = df[df.fold == fold_k].copy()
    train["is_valid"] = 0
    val["is_valid"]   = 1
    return pd.concat([train[["fname","label","is_valid"]],
                      val[["fname","label","is_valid"]]],
                     ignore_index=True)
