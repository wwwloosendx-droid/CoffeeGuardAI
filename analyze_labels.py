import os
from collections import Counter

dataset_path = r"C:\Users\hp\Desktop\Coffee and Cashew Nut Dataset\Coffee\Coffee"

class_counts = Counter()

for batch in os.listdir(dataset_path):
    batch_path = os.path.join(dataset_path, batch)

    images_path = os.path.join(batch_path, "images")
    labels_path = os.path.join(batch_path, "labels")

    if not os.path.exists(labels_path):
        continue

    for label_file in os.listdir(labels_path):
        file_path = os.path.join(labels_path, label_file)

        try:
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = parts[0]
                        class_counts[class_id] += 1
        except:
            pass

print("\nCLASS DISTRIBUTION:")
for k, v in class_counts.items():
    print(f"Class {k}: {v} samples")