"""
Day 1 — Explore class distribution and build train/val/test splits.

Two things matter more here than in a typical Kaggle split:

1. HAM10000 has multiple images per *lesion* (lesion_id), not per image —
   the same mole photographed at different zoom/angle. Splitting by image_id
   naively leaks near-duplicate lesions across train/val/test and inflates
   your validation numbers. We split by lesion_id instead.

2. The classes are heavily imbalanced (nv ~67%, mel ~11%, and some classes
   under 2%). We report that plainly and carry class weights forward into
   training rather than pretending accuracy alone means anything here.

Usage:
    python data/prepare_splits.py
Outputs:
    data_processed/splits/{train,val,test}.csv
    data_processed/class_distribution.png
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, SPLITS_DIR, PROCESSED_DIR, SEED  # noqa: E402


def load_metadata() -> pd.DataFrame:
    meta_path = DATA_DIR / "HAM10000_metadata.csv"
    if not meta_path.exists():
        sys.exit(
            f"Metadata not found at {meta_path}.\n"
            "Run data/download_data.py first."
        )
    df = pd.read_csv(meta_path)
    df["image_path"] = df["image_id"].apply(lambda x: str(DATA_DIR / "images" / f"{x}.jpg"))
    return df


def report_class_distribution(df: pd.DataFrame):
    counts = df["dx"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    print("\n--- Class distribution (by image, not lesion) ---")
    for cls in counts.index:
        print(f"  {cls:6s}  n={counts[cls]:5d}  ({pct[cls]}%)")
    print(f"  {'TOTAL':6s}  n={len(df)}")
    print(
        "\nNote the imbalance: nv dominates, mel/bcc/akiec (the classes we most "
        "need recall on) are minorities. This is exactly why Day 2 uses a "
        "weighted loss instead of relying on raw accuracy.\n"
    )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        counts.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_xlabel("Image count")
        ax.set_title("HAM10000 class distribution (imbalanced)")
        fig.tight_layout()
        out_path = PROCESSED_DIR / "class_distribution.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved distribution chart to {out_path}")
    except ImportError:
        print("matplotlib not installed — skipping chart, counts above are still valid.")


def lesion_level_split(df: pd.DataFrame, seed: int = SEED):
    """Group by lesion_id so the same physical lesion never appears in two splits."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["lesion_id"]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["lesion_id"]))
    val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

    return train_df, val_df, test_df


def assert_no_leakage(train_df, val_df, test_df):
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    assert not (train_lesions & val_lesions), "Leakage between train/val!"
    assert not (train_lesions & test_lesions), "Leakage between train/test!"
    assert not (val_lesions & test_lesions), "Leakage between val/test!"
    print("Leakage check passed: no lesion_id appears in more than one split.")


def main():
    df = load_metadata()
    report_class_distribution(df)

    train_df, val_df, test_df = lesion_level_split(df)
    assert_no_leakage(train_df, val_df, test_df)

    print("\n--- Split sizes (by image) ---")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:5s}: {len(split_df)} images, {split_df['lesion_id'].nunique()} lesions")
        split_df.to_csv(SPLITS_DIR / f"{name}.csv", index=False)

    print(f"\nSplits written to {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
