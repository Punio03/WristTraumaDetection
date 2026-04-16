import pathlib
import wandb
import yaml
import sys
import argparse
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / 'data/dataset/data.yaml'
CONFIG = ROOT / 'config'
RUNS = ROOT / 'runs'

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
args = parser.parse_args()

config_path = CONFIG / args.config

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        model_config = config['model']
except FileNotFoundError as e:
    print('File not found!', file=sys.stderr, flush=True)
    exit(1)

model = YOLO(model_config['weights'])

model.train(
    # -- Training Configuration --
    epochs=model_config.get('epochs', 100),
    batch=model_config.get('batch', 16),
    patience=model_config.get('patience', 50),
    device=model_config.get('device', None),
    lr0=model_config.get('lr0', 0.01),
    optimizer=model_config.get('optimizer', 'auto'),
    imgsz=model_config.get('imgsz', 640),

    # -- Project Specific --
    data=str(DATA),
    project=config['project_name'],
    name=config['run_name'],
    plots=config.get('plots', False),
    save=config.get('save', True),

    # -- Augmentations --
    mosaic=model_config.get('mosaic', 1.0),
    fliplr=model_config.get('fliplr', 0.5),
    flipud=model_config.get('flipud', 0.0),
    degrees=model_config.get('degrees', 0.0),
    mixup=model_config.get('mixup', 0.0),
    translate=model_config.get('translate', 0.1),
    scale=model_config.get('scale', 0.5),
    shear=model_config.get('shear', 0.0)
)

wandb.finish()
