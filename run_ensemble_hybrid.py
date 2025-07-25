# run_ensemble_hybrid.py

import os
from tqdm import tqdm
from torchvision import transforms
from src.models import load_yolo_model, load_resnet18, run_ensemble_pipeline
from src.utils import evaluate_detections

# --- Config ---
YOLO_WEIGHTS = 'weights/yolov8n_v3.6.pt'
RESNET_WEIGHTS = 'weights/resnet18.pth'
TEST_IMAGE_DIR = 'datasetv3/test/images'
TEST_GT_DIR = 'datasetv3/test/labels'
DEVICE = 'cpu'

# --- Simplified Ensemble Hyperparameters ---
ENSEMBLE_WEIGHTS = {'yolo': 0.7, 'resnet': 0.3}
HIGH_CONF_THRESHOLD = 0.5  # Detections above this are trusted. Below this, they are sent to the ensemble.
FINAL_THRESHOLD = 0.3      # The final hurdle for the combined score.

def main():
    print("--- Running Hybrid Model (Ensemble Method) ---")
    yolo_model = load_yolo_model(YOLO_WEIGHTS)
    resnet_model = load_resnet18(RESNET_WEIGHTS, device=DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    all_predictions = {}
    image_files = os.listdir(TEST_IMAGE_DIR)
    
    for img_file in tqdm(image_files, desc="Running Ensemble Pipeline"):
        img_path = os.path.join(TEST_IMAGE_DIR, img_file)
        
        # Call the modified function with the simplified parameters
        all_predictions[img_file] = run_ensemble_pipeline(
            yolo_model=yolo_model, 
            resnet_model=resnet_model, 
            image_path=img_path, 
            transform=transform, 
            weights=ENSEMBLE_WEIGHTS, 
            high_conf_thresh=HIGH_CONF_THRESHOLD, 
            final_thresh=FINAL_THRESHOLD, 
            device=DEVICE
        )
        output_path = os.path.join("results/ensemble_inf", img_file)
        
    evaluate_detections(all_predictions, TEST_GT_DIR, TEST_IMAGE_DIR, results_prefix="hybrid_ensemble")

if __name__ == "__main__":
    main()