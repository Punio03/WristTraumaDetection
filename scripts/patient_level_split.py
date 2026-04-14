import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

DATA_PATH = Path(__file__).parent.parent / "data"
DATASET_CSV = DATA_PATH / "dataset.csv"
ANN_DIR = DATA_PATH / "folder_structure/supervisely/wrist/ann"
NEW_DATASET_CSV = DATA_PATH / "dataset_with_split.csv"
PATIENT_SPLIT_CSV = DATA_PATH / "patient_split.csv"

SPLITS = ("train", "val", "test")
CLASSES = [
    "boneanomaly",
    "bonelesion",
    "foreignbody",
    "fracture",
    "metal",
    "periostealreaction",
    "pronatorsign",
    "softtissue",
]


@dataclass
class Patient:
    patient_id: int | str
    num_images: int = 0
    has_boneanomaly: bool = False
    has_bonelesion: bool = False
    has_foreignbody: bool = False
    has_fracture: bool = False
    has_metal: bool = False
    has_periostealreaction: bool = False
    has_pronatorsign: bool = False
    has_softtissue: bool = False


def get_classes_for_image(filestem: str):
    ann_path = ANN_DIR / f"{filestem}.json"
    data = json.loads(ann_path.read_text())

    object_classes = sorted(
        {
            obj["classTitle"]
            for obj in data.get("objects", [])
            if obj["classTitle"] not in ("axis", "text")
        }
    )

    return object_classes


def get_patient_classes():
    df = pd.read_csv(DATASET_CSV)

    patient_classes = {}
    for _, row in df.iterrows():
        filestem = row["filestem"]
        patient_id = row["patient_id"]

        if patient_id not in patient_classes:
            patient_classes[patient_id] = Patient(patient_id)

        classes = get_classes_for_image(filestem)
        for cls in classes:
            setattr(patient_classes[patient_id], f"has_{cls.lower()}", True)

        patient_classes[patient_id].num_images += 1

    return list(patient_classes.values())


def patient_level_split(
    patients: list[Patient],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
):
    ratios = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }

    def largest_remainder_targets(total: int) -> dict[str, int]:
        raw = {split: total * ratios[split] for split in SPLITS}
        floors = {split: math.floor(value) for split, value in raw.items()}
        remainder = total - sum(floors.values())

        order = sorted(
            SPLITS,
            key=lambda split: (raw[split] - floors[split], -SPLITS.index(split)),
            reverse=True,
        )

        for split in order[:remainder]:
            floors[split] += 1

        return {split: int(value) for split, value in floors.items()}

    patient_targets = largest_remainder_targets(len(patients))
    image_targets = largest_remainder_targets(sum(p.num_images for p in patients))

    class_targets = {}
    for cls in CLASSES:
        class_targets[cls] = largest_remainder_targets(
            sum(int(getattr(p, f"has_{cls}")) for p in patients)
        )

    class_weights = {}
    for cls in CLASSES:
        total_positive = sum(int(getattr(p, f"has_{cls}")) for p in patients)
        class_weights[cls] = max(1, math.ceil(1000 / max(1, total_positive)))

    model = cp_model.CpModel()

    x = {}
    for i, patient in enumerate(patients):
        for split in SPLITS:
            x[i, split] = model.NewBoolVar(f"x_{i}_{split}")

    for i in range(len(patients)):
        model.Add(sum(x[i, split] for split in SPLITS) == 1)

    for split in SPLITS:
        model.Add(
            sum(x[i, split] for i in range(len(patients))) == patient_targets[split]
        )

    objective_terms = []

    total_images = sum(p.num_images for p in patients)
    for split in SPLITS:
        real_images = sum(
            patients[i].num_images * x[i, split] for i in range(len(patients))
        )

        img_dev = model.NewIntVar(0, total_images, f"img_dev_{split}")
        model.Add(img_dev >= real_images - image_targets[split])
        model.Add(img_dev >= image_targets[split] - real_images)

        objective_terms.append(img_dev)

    for cls in CLASSES:
        total_positive = sum(int(getattr(p, f"has_{cls}")) for p in patients)

        for split in SPLITS:
            real_class_count = sum(
                int(getattr(patients[i], f"has_{cls}")) * x[i, split]
                for i in range(len(patients))
            )

            dev = model.NewIntVar(0, total_positive, f"dev_{cls}_{split}")
            model.Add(dev >= real_class_count - class_targets[cls][split])
            model.Add(dev >= class_targets[cls][split] - real_class_count)

            objective_terms.append(class_weights[cls] * 10 * dev)

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 42

    status = solver.Solve(model)

    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"Solver failed: {solver.StatusName(status)}")

    patient_to_split = {}
    for i, patient in enumerate(patients):
        for split in SPLITS:
            if solver.Value(x[i, split]) == 1:
                patient_to_split[patient.patient_id] = split
                break

    return patient_to_split


def validate_split(patients: list[Patient], patient_to_split: dict):
    patient_ids = {p.patient_id for p in patients}

    assert (
        set(patient_to_split.keys()) == patient_ids
    ), "Mismatch between patients and split map."
    assert all(
        split in SPLITS for split in patient_to_split.values()
    ), "Unknown split name found."

    print("\n=== Split sizes ===")
    split_counts = Counter(patient_to_split.values())
    for split in SPLITS:
        print(split, "patients:", split_counts[split])

    print("\n=== Patient-level class balance ===")
    rows = []
    for split in SPLITS:
        split_patients = [
            p for p in patients if patient_to_split[p.patient_id] == split
        ]

        row = {
            "split": split,
            "patients": len(split_patients),
            "images": sum(p.num_images for p in split_patients),
        }

        for cls in CLASSES:
            row[cls] = sum(int(getattr(p, f"has_{cls}")) for p in split_patients)

        rows.append(row)

    balance_df = pd.DataFrame(rows)
    print(balance_df.to_string(index=False))

    print("\n=== Global ratios per class ===")
    total_patients = len(patients)
    total_images = sum(p.num_images for p in patients)
    print("all_patients:", total_patients)
    print("all_images:", total_images)

    for cls in CLASSES:
        total_cls = sum(int(getattr(p, f"has_{cls}")) for p in patients)
        print(cls, "total_positive_patients:", total_cls)


def save_split(patients: list[Patient], patient_to_split: dict):
    df = pd.read_csv(DATASET_CSV)

    df["split"] = df["patient_id"].map(patient_to_split)
    if df["split"].isna().sum() != 0:
        raise ValueError("Some rows in dataset.csv did not receive a split.")

    df.to_csv(NEW_DATASET_CSV, index=False)

    patient_rows = []
    for p in patients:
        patient_rows.append(
            {
                "patient_id": p.patient_id,
                "split": patient_to_split[p.patient_id],
                "num_images": p.num_images,
                "has_boneanomaly": p.has_boneanomaly,
                "has_bonelesion": p.has_bonelesion,
                "has_foreignbody": p.has_foreignbody,
                "has_fracture": p.has_fracture,
                "has_metal": p.has_metal,
                "has_periostealreaction": p.has_periostealreaction,
                "has_pronatorsign": p.has_pronatorsign,
                "has_softtissue": p.has_softtissue,
            }
        )

    patient_df = pd.DataFrame(patient_rows)
    patient_df.to_csv(PATIENT_SPLIT_CSV, index=False)


if __name__ == "__main__":
    patients = get_patient_classes()
    patient_to_split = patient_level_split(patients)
    validate_split(patients, patient_to_split)
    save_split(patients, patient_to_split)
