import argparse
import os
import shutil
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from ultralytics import YOLO


def parse_simple_yaml(path: Path) -> dict:
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if not value:
                continue
            if key in {'train', 'val', 'test'}:
                data[key] = value
            elif key == 'nc':
                try:
                    data[key] = int(value)
                except ValueError:
                    data[key] = value
            elif key == 'names':
                if value.startswith('[') and value.endswith(']'):
                    names = [name.strip().strip("'\"") for name in value[1:-1].split(',')]
                    data[key] = names
                else:
                    data[key] = [value]
    return data


def validate_data_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Data YAML file not found: {path}")
    config = parse_simple_yaml(path)
    missing = [key for key in ('train', 'val', 'nc', 'names') if key not in config]
    if missing:
        raise ValueError(f"data.yaml is missing required keys: {', '.join(missing)}")

    for split in ('train', 'val'):
        if split in config and not Path(config[split]).exists():
            raise FileNotFoundError(f"Path for '{split}' does not exist: {config[split]}")

    return config


def copy_best_weight(source: Path, destination: Path, force: bool = False):
    if not source.exists():
        raise FileNotFoundError(f"Trained weight file not found: {source}")
    if destination.exists() and not force:
        print(f"Destination already exists: {destination}. Use --force to overwrite.")
        return
    shutil.copy2(source, destination)
    print(f"✅ Copied best weights to: {destination}")


def main():
    parser = argparse.ArgumentParser(
        description='Train a YOLOv8 coffee leaf disease detection model and save best.pt in the project root.'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='YOLOv8-Based-SUNet-Real-Time-Coffee-Leaf-Disease-Detection-Using-a-Hybrid-Deep-Learning-Model/data.yaml',
        help='Path to YOLO data config YAML file.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8m.pt',
        help='YOLOv8 backbone to use for training (e.g. yolov8n.pt, yolov8m.pt, yolov8l.pt).'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size for training and validation.')
    parser.add_argument('--device', type=str, default='0', help='Device to use for training (cpu or gpu id).')
    parser.add_argument('--project', type=str, default='runs/train', help='Ultralytics training project folder.')
    parser.add_argument('--name', type=str, default='coffee_guard', help='Ultralytics training run name.')
    parser.add_argument('--save', type=str, default='best.pt', help='Copy best weights to this destination after training.')
    parser.add_argument('--force', action='store_true', help='Overwrite the save destination if it already exists.')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience for learning rate scheduling.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers.')
    args = parser.parse_args()

    data_file = Path(args.data)
    if not data_file.exists():
        raise FileNotFoundError(
            f"The default data config was not found: {data_file}\n"
            "Update the path with --data or place a valid data.yaml at the default location."
        )

    config = validate_data_yaml(data_file)
    print("\n=== YOLO Training Configuration ===")
    print(f"data:      {data_file}")
    print(f"model:     {args.model}")
    print(f"epochs:    {args.epochs}")
    print(f"batch:     {args.batch}")
    print(f"imgsz:     {args.imgsz}")
    print(f"device:    {args.device}")
    print(f"project:   {args.project}")
    print(f"run name:  {args.name}")
    print(f"save path: {args.save}")
    print("===================================\n")

    if torch is not None and torch.cuda.is_available():
        print(f"Using CUDA device: {args.device}")
    else:
        print("CUDA unavailable; training will run on CPU.")

    model = YOLO(args.model)
    model.train(
        data=str(data_file),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        optimizer='Adam',
        lr0=args.lr,
        patience=args.patience,
        augment=True,
        workers=args.workers,
    )

    weights_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    if weights_path.exists():
        destination = Path(args.save)
        copy_best_weight(weights_path, destination, force=args.force)
    else:
        print(f"Training finished, but best weights were not found at: {weights_path}")
        print("Check the training run directory for available weights.")

    print('\nTraining complete.')


if __name__ == '__main__':
    main()
