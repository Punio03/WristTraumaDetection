import os
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data"
DATASET_WITH_SPLIT_CSV = DATA_PATH / "dataset_with_split.csv"
YOLOV5_PATH = DATA_PATH / "folder_structure/yolov5"
LABELS_PATH = YOLOV5_PATH / "labels"
DATASET_PATH = DATA_PATH / "dataset"
IMAGES_PATH = DATASET_PATH / "images"
NEW_LABELS_PATH = DATA_PATH / "dataset" / "labels"
YAML_PATH = DATASET_PATH / "data.yaml"

SPLITS = ("train", "val", "test")


def create_directory_structure():
    for split in SPLITS:
        split_dir = NEW_LABELS_PATH / split
        os.makedirs(split_dir, exist_ok=True)


def move_labels_to_splits():
    df = pd.read_csv(DATASET_WITH_SPLIT_CSV)
    for _, row in df.iterrows():
        filestem = row["filestem"]
        split = row["split"]

        src_path = LABELS_PATH / f"{filestem}.txt"
        dst_path = NEW_LABELS_PATH / split / src_path.name

        if src_path.exists():
            os.rename(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist.")


def filter_labels():
    for split in SPLITS:
        split_dir = NEW_LABELS_PATH / split
        for label_file in split_dir.iterdir():
            if label_file.is_file():
                with label_file.open() as f:
                    lines = f.readlines()
                filtered_lines = [line for line in lines if not line.startswith("8")]
                with label_file.open("w") as f:
                    f.writelines(filtered_lines)


def create_yaml_file():
    yaml_content = f"""
path: {DATASET_PATH}
train: images/train
val: images/val
test: images/test

names:
  0: boneanomaly
  1: bonelesion
  2: foreignbody
  3: fracture
  4: metal
  5: periostealreaction
  6: pronatorsign
  7: softtissue
"""
    with YAML_PATH.open("w") as f:
        f.write(yaml_content)


def delete_empty_directories():
    if not any(LABELS_PATH.iterdir()):
        os.rmdir(LABELS_PATH)

    os.remove(YOLOV5_PATH / "meta.yaml")

    if not any(YOLOV5_PATH.iterdir()):
        os.rmdir(YOLOV5_PATH)


if __name__ == "__main__":
    create_directory_structure()
    move_labels_to_splits()
    filter_labels()
    create_yaml_file()
    delete_empty_directories()
