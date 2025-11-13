import argparse
import csv
import random
import shutil
from pathlib import Path
import pandas as pd

SEED = 42
N_PER_CLASS = 410
random.seed(SEED)

# Paths
CSV_PATH = Path("./dl_morph_labelling/raw_data/summer_hts_data_matt.csv")
BASE_RAW = Path("./dl_morph_labelling/images/raw_images")
MANIFEST_CSV = Path("./dl_morph_labelling/manifests/head4_manifest.csv")
COPY_ROOT = Path("./dl_morph_labelling/images/raw_images/head4_raw")  # only used if --copy

# CSV columns
CSV_FOLDER_COL = "robot_folder"
CSV_LABEL_COL = "robot_morphology"   # free text label per folder
CSV_CRYSTALLINE_COL = "crystalline"  # boolean-ish flag

# C3 sources (non-crystals)
C3_DIRS = {
    "clear": Path("./dl_morph_labelling/C3/clear"),
    "precipitate": Path("./dl_morph_labelling/C3/precipitate"),
    "other": Path("./dl_morph_labelling/C3/other"),
}

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CRYSTAL_MORPHS = {"block", "plate", "needle"}  # normalized set

NORM_MAP = {
    "blocks": "block",
    "blocky": "block",
    "plates": "plate",
    "needles": "needle",
    "crystal": "",           # not a morphology; keep blank
    "crystals": "",          # not a morphology; keep blank
}

def norm_label(x: str) -> str:
    if not isinstance(x, str):
        return ""
    v = x.strip().lower()
    return NORM_MAP.get(v, v)

def truthy(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in {"1","true","yes","y","t"}
    if isinstance(x, (int, float)):
        return x == 1
    if isinstance(x, bool):
        return x
    return False

def list_images(d: Path):
    if not d.exists():
        return []
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXT]

def build_crystal_pool_from_csv():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    for col in (CSV_FOLDER_COL, CSV_LABEL_COL):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {CSV_PATH}. Found: {list(df.columns)}")

    # Normalize morphology text and detect crystals
    df["_morph_norm"] = df[CSV_LABEL_COL].apply(norm_label)
    has_crystal_morph = df["_morph_norm"].isin(CRYSTAL_MORPHS)
    has_crystalline_flag = CSV_CRYSTALLINE_COL in df.columns and df[CSV_CRYSTALLINE_COL].apply(truthy)
    df["_is_crystal"] = has_crystal_morph | has_crystalline_flag

    crystals_df = df[df["_is_crystal"]].copy()
    if crystals_df.empty:
        uniq_vals = sorted(set(df[CSV_LABEL_COL].astype(str).str.strip().str.lower()))
        raise ValueError(
            "No crystal rows found. Check robot_morphology/crystalline values.\n"
            f"Unique robot_morphology values (normalized): {uniq_vals}"
        )

    # For folders with a known crystal morphology, take the mode; else leave blank
    morph_by_folder = (
        df[df["_morph_norm"].isin(CRYSTAL_MORPHS)]
        .groupby(CSV_FOLDER_COL)["_morph_norm"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
        .to_dict()
    )

    pool = []
    for folder in crystals_df[CSV_FOLDER_COL].astype(str).unique():
        folder_path = BASE_RAW / folder
        imgs = list_images(folder_path)
        if not imgs:
            continue
        morph = morph_by_folder.get(folder, "")  # blank if unknown
        for p in imgs:
            pool.append((p, morph))  # (Path, crystal_morphology or "")

    # Deduplicate
    seen = set()
    uniq = []
    for p, m in pool:
        if p not in seen and p.exists():
            uniq.append((p, m))
            seen.add(p)
    return uniq

def sample_items(pool, n):
    if len(pool) < n:
        raise ValueError(f"Not enough items to sample {n}. Found {len(pool)}.")
    return random.sample(pool, n)

def copy_if_needed(src_path: Path, label: str, do_copy: bool):
    if not do_copy:
        return src_path.as_posix()
    dest_dir = COPY_ROOT / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / src_path.name
    shutil.copy2(src_path, dst)
    return dst.as_posix()

def main(do_copy: bool, n_per_class: int):
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # Crystals from robot CSV (per-folder -> images)
    crystal_pool = build_crystal_pool_from_csv()  # list of (Path, crystal_morph)
    crystal_sel = sample_items(crystal_pool, n_per_class)
    for p, morph in crystal_sel:
        rows.append([copy_if_needed(p, "crystal", do_copy), "crystal", morph])

    # Non-crystals from C3: clear, precipitate, other
    for label, src in C3_DIRS.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing C3 source dir: {src}")
        pool = list_images(src)
        sel = sample_items(pool, n_per_class)
        for p in sel:
            rows.append([copy_if_needed(p, label, do_copy), label, ""])

    # Write manifest
    with MANIFEST_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "label", "crystal_morphology"])
        w.writerows(rows)

    print(f"Wrote manifest: {MANIFEST_CSV}")
    if do_copy:
        print(f"Copied files into: {COPY_ROOT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true", help="Copy sampled files into head4_raw/<class>")
    parser.add_argument("--n", type=int, default=N_PER_CLASS, help="Samples per class (default 410)")
    args = parser.parse_args()
    main(do_copy=args.copy, n_per_class=args.n)