import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"

SCRIPTS = [
    ROOT / "scripts" / "patient_level_split.py",
    ROOT / "scripts" / "move_images_to_splits.py",
    ROOT / "scripts" / "prepare_yolo_dataset.py",
]


def run_script(script_path: Path):
    print(f"\n=== Running: {script_path.name} ===")
    result = subprocess.run([str(PYTHON), str(script_path)], cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{script_path.name} failed with code {result.returncode}")


def main():
    for script in SCRIPTS:
        run_script(script)

    print("\nDataset pipeline finished successfully.")


if __name__ == "__main__":
    main()
