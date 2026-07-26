# Training the CoffeeLeaf YOLO Model

This file explains how to train a YOLOv8 model for coffee leaf disease detection using the new `train_yolo.py` script.

## Requirements

- `python` installed
- packages from `requirements.txt`
- more disk space for `runs/train`
- a valid `data.yaml` file pointing to your dataset

## Recommended command

```bash
python train_yolo.py \
  --data YOLOv8-Based-SUNet-Real-Time-Coffee-Leaf-Disease-Detection-Using-a-Hybrid-Deep-Learning-Model/data.yaml \
  --model yolov8m.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 1280 \
  --device 0
```

## Notes

- If you do not have a GPU, set `--device cpu`.
- If `best.pt` already exists, add `--force` to overwrite it.
- The script will copy the best weights into the repo as `best.pt` after training.

## If your dataset path is wrong

Edit the `data.yaml` file and update it to use a local dataset folder inside the nested training directory. For example, if your dataset files are stored in:

- `YOLOv8-Based-SUNet-Real-Time-Coffee-Leaf-Disease-Detection-Using-a-Hybrid-Deep-Learning-Model/Dataset/train`
- `YOLOv8-Based-SUNet-Real-Time-Coffee-Leaf-Disease-Detection-Using-a-Hybrid-Deep-Learning-Model/Dataset/valid`
- `YOLOv8-Based-SUNet-Real-Time-Coffee-Leaf-Disease-Detection-Using-a-Hybrid-Deep-Learning-Model/Dataset/test`

use:

```yaml
path: Dataset
train: train
val: valid
test: test
nc: 4
names: ['brown_eye_spot', 'leaf_miner', 'leaf_rust', 'red_spider_mite']
```

If your dataset lives elsewhere, set `path:` to the base folder and use relative `train`/`val`/`test` paths from that location.

## Validation

After training, you can validate the model with YOLO commands or by using the app's normal inference.

```bash
python -m ultralytics.yolo.val model=best.pt data=YOLOv8-Based-SUNet-Real-Time-Coffee-Leaf-Disease-Detection-Using-a-Hybrid-Deep-Learning-Model/data.yaml imgsz=1280
```
