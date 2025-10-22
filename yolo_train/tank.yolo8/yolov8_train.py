from ultralytics import YOLO
import os
import glob
from pathlib import Path

# Get the script's directory (tank.yolo8/)
SCRIPT_DIR = Path(__file__).parent.resolve()
# Project root is one level up from script directory
PROJECT_ROOT = SCRIPT_DIR.parent

# Dataset paths relative to project root
TANK_ONLY_DATA = PROJECT_ROOT / "dataset_only_tank" / "data.yaml"

print(f"Script directory: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data YAML: {TANK_ONLY_DATA.resolve()}")

# Training 1: Tank only dataset (class ID = 0)
print("\n" + "="*60)
print("TRAINING 1: Tank Only Dataset")
print("="*60)

model = YOLO("yolov8n.pt")
print(f"\nStarting training with {TANK_ONLY_DATA}...")
model.train(
    data=str(TANK_ONLY_DATA),
    epochs=10,
    imgsz=640,
    project=str(PROJECT_ROOT / "runs" / "detect"),
    name="train_only_tank",
    exist_ok=True  # Overwrite existing results
)
metrics = model.val()
print("\nTraining 1 Results:")
print(f"  mAP50: {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")
print("="*60)

