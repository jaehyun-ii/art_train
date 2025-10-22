from ultralytics import YOLO
import os
import glob
from pathlib import Path

# Get the script's directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Dataset paths relative to project root
TANK_ONLY_DATA = PROJECT_ROOT / "dataset_only_tank" / "data.yaml"
MERGED_DATA = PROJECT_ROOT / "merged_dataset" / "data.yaml"

# Training 1: Tank only dataset (class ID = 0)
print("\n" + "="*60)
print("TRAINING 1: Tank Only Dataset")
print("="*60)

model = YOLO("yolov8n.pt")
print(f"\nStarting training with {TANK_ONLY_DATA}...")
model.train(
    data=str(TANK_ONLY_DATA),
    epochs=100,
    imgsz=640,
    project="runs/detect",
    name="train_tank_only",
    exist_ok=True  # Overwrite existing results
)
metrics = model.val()
print("\nTraining 1 Results:")
print(f"  mAP50: {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")
print("="*60)


# Training 2: Tank with COCO classes (class ID = 80, frozen backbone)
print("\n" + "="*60)
print("TRAINING 2: Tank + COCO Dataset (Frozen Backbone)")
print("="*60)

model2 = YOLO("yolov8n.pt")
print(f"\nStarting training with {MERGED_DATA}...")
model2.train(
    data=str(MERGED_DATA),
    epochs=100,
    imgsz=640,
    project="runs/detect",
    name="train_with_coco",
    exist_ok=True  # Overwrite existing results
)
metrics2 = model2.val()
print("\nTraining 2 Results:")
print(f"  mAP50: {metrics2.box.map50:.4f}")
print(f"  mAP50-95: {metrics2.box.map:.4f}")
print("="*60)

print("\n" + "="*60)
print("ALL TRAINING COMPLETED")
print("="*60)
print(f"Model 1 (Tank only): runs/detect/train_tank_only/weights/best.pt")
print(f"Model 2 (Tank+COCO): runs/detect/train_with_coco/weights/best.pt")
print("="*60)

