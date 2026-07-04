import os

dataset_path = r"C:\Users\hp\Desktop\Coffee and Cashew Nut Dataset\Coffee\Coffee"

shown = {}

for batch in os.listdir(dataset_path):
    labels_path = os.path.join(dataset_path, batch, "labels")

    if not os.path.exists(labels_path):
        continue

    for file in os.listdir(labels_path):
        path = os.path.join(labels_path, file)

        with open(path, "r") as f:
            line = f.readline().strip()
            if line:
                cls = line.split()[0]

                if cls not in shown:
                    shown[cls] = path

# show ONE example file per class
for k, v in shown.items():
    print("\nCLASS:", k)
    print("Example file:", v)