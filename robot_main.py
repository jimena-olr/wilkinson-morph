import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dl_morph_labelling.preprocessing import (
    preprocess_crystal_dir,
    preprocess_noncrystal_dir_canny,
    build_head1_manifest,
)
from dl_morph_labelling.model import robot_kfold_fastai


def run_head1_binary(out_size=(224, 224), n_splits=5, test_size=0.2, seed=42):
    """
    Train the head1 binary classifier: crystal vs non-crystal.

    Pipeline:
      1. Preprocess crystal and non-crystal images into standardized 224x224 PNGs.
      2. Build a balanced manifest of (fname, label) where label=1 (crystal), 0 (non-crystal).
      3. Split that manifest into an 80/20 stratified train/test split.
      4. Run k-fold CV (on the 80% train+val pool only) via robot_kfold_fastai.
      5. Save train/test manifests for later evaluation.
    """
    base = Path("./dl_morph_labelling")

    # --- 1) Define raw and preprocessed directories ---
    crystal_raw = base / "images" / "raw_images"

    c3_dir = base / "C3"
    noncrystal_sources = []
    for name in ("clear", "other", "precipitate"):
        p = c3_dir / name
        if p.exists() and p.is_dir():
            noncrystal_sources.append(p)

    # Legacy fallback, if you ever used noncrystals_raw
    fallback = c3_dir / "noncrystals_raw"
    if not noncrystal_sources and fallback.exists() and fallback.is_dir():
        noncrystal_sources.append(fallback)

    # Last resort debug fallback – treat raw crystals also as "non-crystals"
    # (only for debugging when nothing else exists)
    if not noncrystal_sources:
        noncrystal_sources.append(crystal_raw)

    crystal_pp = base / "images" / "preprocessed" / "crystal"
    noncrystal_pp = base / "images" / "preprocessed" / "noncrystal"

    # Ensure preprocessed directories exist; clear noncrystal_pp so it does not accumulate stale images
    crystal_pp.mkdir(parents=True, exist_ok=True)
    if noncrystal_pp.exists():
        for f in noncrystal_pp.iterdir():
            try:
                if f.is_file():
                    f.unlink()
            except Exception:
                # ignore read-only / transient issues
                pass
    noncrystal_pp.mkdir(parents=True, exist_ok=True)

    # --- 2) Preprocess images ---
    print("[Head1] Preprocessing crystals …")
    preprocess_crystal_dir(crystal_raw, crystal_pp, out_size)

    print("[Head1] Preprocessing non-crystals (Canny) from:", noncrystal_sources)
    for src in noncrystal_sources:
        preprocess_noncrystal_dir_canny(src, noncrystal_pp, out_size)

    # --- 3) Build balanced manifest over preprocessed images ---
    manifests_dir = base / "manifests"
    manifest = manifests_dir / "head1_manifest.csv"
    print("[Head1] Building balanced manifest …")
    build_head1_manifest(crystal_pp, noncrystal_pp, manifest, balance=True, seed=seed)

    print("[Head1] Loading manifest for 80/20 split …")
    full_df = pd.read_csv(manifest)[["fname", "label"]]

    # --- 4) 80/20 stratified train/test split ---
    print("[Head1] Creating stratified train/test split (test_size = {:.0%}) …".format(test_size))
    trainval_df, test_df = train_test_split(
        full_df,
        test_size=test_size,
        stratify=full_df["label"],
        random_state=seed,
    )

    # Save splits for reproducibility and later test-time evaluation
    manifests_dir.mkdir(parents=True, exist_ok=True)
    trainval_path = manifests_dir / "head1_trainval.csv"
    test_path = manifests_dir / "head1_test.csv"
    trainval_df.to_csv(trainval_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[Head1] Train/val size: {len(trainval_df)}  → {trainval_path}")
    print(f"[Head1] Test size:      {len(test_df)}  → {test_path}")

    # --- 5) Run k-fold CV on the 80% train+val pool only ---
    # robot_kfold_fastai expects a df with columns ['fname', 'label']
    print("[Head1] Starting k-fold training on train/val pool …")
    robot_kfold_fastai(trainval_df[["fname", "label"]], n_splits=n_splits)
    print("[Head1] Finished k-fold training. Test set is frozen on disk; evaluate separately when ready.")


def run_robot_original(n_splits=5):
    """
    Original robot labelling pipeline from the upstream repo.

    This uses the pre-made robot labelling CSV via get_robot_labelling_df
    and runs robot_kfold_fastai on that dataset.
    """
    from dl_morph_labelling.preprocessing import get_robot_labelling_df

    print("[Robot] Loading original robot labelling dataset …")
    robot_df = get_robot_labelling_df()
    print(f"[Robot] Dataset size: {len(robot_df)}")
    robot_kfold_fastai(robot_df, n_splits=n_splits)
    print("[Robot] Finished k-fold training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wilkinson lab morphology models")
    parser.add_argument(
        "--mode",
        choices=["head1", "robot"],
        default="head1",
        help="head1 = crystal vs non-crystal (new pipeline); robot = original robot labelling pipeline",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of CV folds for training (default: 5).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data reserved as held-out test set for head1 (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42).",
    )
    args = parser.parse_args()

    if args.mode == "head1":
        run_head1_binary(n_splits=args.splits, test_size=args.test_size, seed=args.seed)
    else:
        run_robot_original(n_splits=args.splits)
