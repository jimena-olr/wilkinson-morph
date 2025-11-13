# --- extend robot_main.py ---
import pandas as pd
import argparse
from pathlib import Path
from dl_morph_labelling.preprocessing import (
    preprocess_crystal_dir, preprocess_noncrystal_dir_canny,
    build_head1_manifest, add_stratified_folds, df_from_manifest_fold
)
from dl_morph_labelling.model import robot_kfold_fastai  # k-fold trainer you already have
# uses get_robot_labelling_df in other modes (left intact) :contentReference[oaicite:4]{index=4}

# ...existing code...
def run_head1_binary(out_size=(224,224), n_splits=5):
    base = Path("./dl_morph_labelling")

    # all raw molecule folders (crystals)
    crystal_raw = base / "images" / "raw_images"

    # collect C3 non-crystal source folders if present
    c3_dir = base / "C3"
    noncrystal_sources = []
    for name in ("clear", "other", "precipitate"):
        p = c3_dir / name
        if p.exists() and p.is_dir():
            noncrystal_sources.append(p)
    # legacy fallbacks
    fallback = c3_dir / "noncrystals_raw"
    if not noncrystal_sources and fallback.exists() and fallback.is_dir():
        noncrystal_sources.append(fallback)
    if not noncrystal_sources:
        # last resort: use raw_images as a rough non-crystal source for debugging
        noncrystal_sources.append(crystal_raw)

    crystal_pp = base / "images" / "preprocessed" / "crystal"
    noncrystal_pp = base / "images" / "preprocessed" / "noncrystal"

    # ensure clean preprocessed dirs for this run (remove/replace if you prefer)
    crystal_pp.mkdir(parents=True, exist_ok=True)
    if noncrystal_pp.exists():
        # clear old noncrystal preprocessed outputs so new ones don't mix
        for f in noncrystal_pp.iterdir():
            try:
                if f.is_file():
                    f.unlink()
                else:
                    # optionally remove subfolders
                    pass
            except Exception:
                pass
    noncrystal_pp.mkdir(parents=True, exist_ok=True)

    print("[Head1] Preprocessing crystals …")
    preprocess_crystal_dir(crystal_raw, crystal_pp, out_size)

    print("[Head1] Preprocessing non-crystals (Canny) from:", noncrystal_sources)
    for src in noncrystal_sources:
        preprocess_noncrystal_dir_canny(src, noncrystal_pp, out_size)

    manifest = base / "manifests" / "head1_manifest.csv"
    print("[Head1] Building balanced manifest …")
    build_head1_manifest(crystal_pp, noncrystal_pp, manifest, balance=True, seed=42)

    print("[Head1] Adding stratified folds …")
    fold_csv = add_stratified_folds(manifest, n_splits=n_splits, seed=42)

    df = pd.read_csv(fold_csv)[["fname", "label"]]
    robot_kfold_fastai(df, n_splits=n_splits)
# ...existing code...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["head1","robot"], default="head1",
                        help="head1 = crystal vs non-crystal; robot = original pipeline")
    args_cli = parser.parse_args()

    if args_cli.mode == "head1":
        run_head1_binary()
    else:
        # your original flow:
        # robot_df = get_robot_labelling_df(); robot_kfold_fastai(robot_df, n_splits=5)
        from dl_morph_labelling.preprocessing import get_robot_labelling_df
        robot_df = get_robot_labelling_df()  # original CSV → df loader :contentReference[oaicite:6]{index=6}
        robot_kfold_fastai(robot_df, n_splits=5)   # existing trainer :contentReference[oaicite:7]{index=7}
