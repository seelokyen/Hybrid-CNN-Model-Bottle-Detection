from ultralytics import YOLO
from src.models import load_yolo_model

# Configurations
MODEL_PATH = 'weights/yolov8n_v3.6.pt'
DATA_YAML = 'unseen_basic/data.yaml'

def main():
    print("--- Evaluating YOLOv8 Baseline Model ---")
    model = load_yolo_model(MODEL_PATH)

    metrics = model.val(
        data=DATA_YAML,
        split='test',
        name='baseline_metrics'
    )
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()