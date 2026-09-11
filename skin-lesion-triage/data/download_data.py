"""
Day 1 — Download HAM10000.

This project intentionally does NOT bundle the dataset (10,015 images,
~2.6 GB, and it isn't ours to redistribute). Run this once, locally or in
Colab, before anything else.

Two supported sources, either works:

1) Kaggle (recommended — simplest auth):
   pip install kaggle
   # place your kaggle.json API token at ~/.kaggle/kaggle.json
   python data/download_data.py --source kaggle

2) Harvard Dataverse (original source, no Kaggle account needed):
   python data/download_data.py --source dataverse

Both paths leave you with this layout under config.DATA_DIR:

    data_raw/
        HAM10000_metadata.csv
        images/
            ISIC_0024306.jpg
            ISIC_0024307.jpg
            ...
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR  # noqa: E402

KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

DATAVERSE_FILES = {
    # DOI: 10.7910/DVN/DBW86T — "Skin Cancer MNIST: HAM10000"
    "HAM10000_metadata.csv": "https://dataverse.harvard.edu/api/access/datafile/3172779",
    "HAM10000_images_part_1.zip": "https://dataverse.harvard.edu/api/access/datafile/3172741",
    "HAM10000_images_part_2.zip": "https://dataverse.harvard.edu/api/access/datafile/3172742",
}


def download_kaggle():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        sys.exit(
            "kaggle package not installed. Run: pip install kaggle --break-system-packages\n"
            "Then place your API token at ~/.kaggle/kaggle.json "
            "(Kaggle account -> Settings -> Create New API Token)."
        )
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)
    _consolidate_kaggle_layout()


def _consolidate_kaggle_layout():
    """The Kaggle mirror ships two image folders (HAM10000_images_part_1/2)
    and metadata as HAM10000_metadata.csv — merge images into one folder."""
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(exist_ok=True)
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        part_dir = DATA_DIR / part
        if part_dir.exists():
            for img in part_dir.glob("*.jpg"):
                shutil.copy(img, images_dir / img.name)
            shutil.rmtree(part_dir)
    print(f"Consolidated {len(list(images_dir.glob('*.jpg')))} images into {images_dir}")


def download_dataverse():
    import urllib.request

    images_dir = DATA_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    for fname, url in DATAVERSE_FILES.items():
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"{fname} already present, skipping download.")
        else:
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, dest)

        if fname.endswith(".zip"):
            print(f"Extracting {fname} ...")
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(images_dir)
            dest.unlink()

    # Dataverse zips sometimes nest images one level deep — flatten them.
    for nested in images_dir.rglob("*.jpg"):
        if nested.parent != images_dir:
            shutil.move(str(nested), str(images_dir / nested.name))
    print(f"Done. {len(list(images_dir.glob('*.jpg')))} images in {images_dir}")


def verify():
    meta = DATA_DIR / "HAM10000_metadata.csv"
    images_dir = DATA_DIR / "images"
    n_images = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    print("\n--- Verification ---")
    print(f"metadata csv found: {meta.exists()}")
    print(f"image count: {n_images} (expect 10015)")
    if not meta.exists() or n_images < 10000:
        print("WARNING: download looks incomplete. Re-run or check your connection.")
    else:
        print("Looks good — proceed to data/prepare_splits.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["kaggle", "dataverse"], default="kaggle")
    args = parser.parse_args()

    if args.source == "kaggle":
        download_kaggle()
    else:
        download_dataverse()
    verify()
