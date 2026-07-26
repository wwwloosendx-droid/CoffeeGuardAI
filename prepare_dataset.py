import argparse
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
LABEL_EXTENSION = '.txt'

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_label_file(path: Path) -> bool:
    return path.suffix.lower() == LABEL_EXTENSION


def ensure_dataset_structure(base_dir: Path) -> Path:
    dataset_dir = base_dir / 'Dataset'
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f'Created dataset base directory: {dataset_dir}')
    for split in ('train', 'valid', 'test'):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        gitkeep = split_dir / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.write_text('')
    return dataset_dir


def copy_files(source: Path, target: Path, force: bool = False) -> int:
    if not source.exists():
        return 0
    count = 0
    for path in sorted(source.iterdir()):
        if path.is_file() and (is_image_file(path) or is_label_file(path)):
            target_path = target / path.name
            if target_path.exists() and not force:
                continue
            shutil.copy2(path, target_path)
            count += 1
    return count


def scan_directory(path: Path) -> tuple[int, int]:
    image_count = 0
    label_count = 0
    if not path.exists():
        return 0, 0
    for file in path.rglob('*'):
        if file.is_file():
            if is_image_file(file):
                image_count += 1
            elif is_label_file(file):
                label_count += 1
    return image_count, label_count


def print_dataset_summary(dataset_dir: Path) -> None:
    print(f'Verifying dataset structure under: {dataset_dir}')
    rows = []
    total_images = 0
    total_labels = 0
    for split in ('train', 'valid', 'test'):
        split_dir = dataset_dir / split
        images, labels = scan_directory(split_dir)
        rows.append((split, images, labels, split_dir.exists()))
        total_images += images
        total_labels += labels

    for split, images, labels, exists in rows:
        status = 'OK' if exists else 'missing'
        print(f'  {split:5s}: {images:5d} images, {labels:5d} labels  [{status}]')
    print(f'  total: {total_images} images, {total_labels} labels')

    if total_images == 0:
        print('\nWarning: no images found in the dataset. Add images and matching YOLO label files to continue.')
    if total_labels == 0:
        print('\nWarning: no label files found. YOLO training requires .txt label files for each image.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Prepare or validate the YOLO dataset folder structure for CoffeeGuardAI training.'
    )
    parser.add_argument(
        '--base',
        type=str,
        default='YOLOv8-Based-SUNet-Real-Time-Coffee-Leaf-Disease-Detection-Using-a-Hybrid-Deep-Learning-Model',
        help='Root folder where Dataset/ train/ valid/ test/ should be created.',
    )
    parser.add_argument(
        '--copy-from',
        type=str,
        default=None,
        help='Optional existing source folder containing train/valid/test subfolders to copy into the expected Dataset layout.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite copied files in the target dataset folders when using --copy-from.',
    )
    args = parser.parse_args()

    base_dir = Path(args.base).resolve()
    dataset_dir = ensure_dataset_structure(base_dir)

    if args.copy_from:
        source_dir = Path(args.copy_from).resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f'Source directory does not exist: {source_dir}')

        copied = 0
        for split in ('train', 'valid', 'test'):
            source_split = source_dir / split
            target_split = dataset_dir / split
            if source_split.exists():
                copied += copy_files(source_split, target_split, force=args.force)
                print(f'Copied {split} files from {source_split} to {target_split}')
            else:
                print(f'No source split found at: {source_split}')
        if copied == 0:
            print('No files were copied. Verify that the source folder contains train/valid/test subdirectories.')
        else:
            print(f'Total files copied: {copied}')

    print_dataset_summary(dataset_dir)
    print('\nDone. You can now add your images and labels inside the Dataset/train, Dataset/valid, and Dataset/test folders.')

    print('To train, run:')
    print('  python train_yolo.py --data YOLOv8-Based-SUNet-Real-Time-Coffee-Leaf-Disease-Detection-Using-a-Hybrid-Deep-Learning-Model/data.yaml --device 0')


if __name__ == '__main__':
    main()
