from ultralytics import YOLO

model = YOLO("best.pt")

print("MODEL LOADED SUCCESSFULLY ✅")
print("Classes detected by this model:")
print(model.names)