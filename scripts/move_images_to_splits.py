import os
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data"
DATASET_WITH_SPLIT_CSV = DATA_PATH / "dataset_with_split.csv"
IMAGES_PATHS = [
    DATA_PATH / "images_part1",
    DATA_PATH / "images_part2",
    DATA_PATH / "images_part3",
    DATA_PATH / "images_part4",
]
NEW_IMAGES_PATH = DATA_PATH / "dataset" / "images"


def create_directory_structure():
    for split in ("train", "val", "test"):
        split_dir = NEW_IMAGES_PATH / split
        os.makedirs(split_dir, exist_ok=True)


def move_images_to_splits():
    image_map = {}
    for image_dir in IMAGES_PATHS:
        for path in image_dir.iterdir():
            if path.is_file():
                image_map[path.stem] = path

    df = pd.read_csv(DATASET_WITH_SPLIT_CSV)
    for _, row in df.iterrows():
        filestem = row["filestem"]
        split = row["split"]

        src_path = image_map[filestem]
        dst_path = NEW_IMAGES_PATH / split / src_path.name

        if src_path.exists():
            os.rename(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist.")


def delete_empty_directories():
    for image_dir in IMAGES_PATHS:
        if not any(image_dir.iterdir()):
            os.rmdir(image_dir)


if __name__ == "__main__":
    create_directory_structure()
    move_images_to_splits()
    delete_empty_directories()
