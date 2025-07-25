import os
from tqdm import tqdm
from torchvision import transforms
from src.models import load_yolo_model, load_resnet18, run_hard_gate_pipeline
from src.utils import evaluate_detections, draw_hard_gate_visualization

# Configurations
YOLO_WEIGHTS = 'weights/yolov8n_v3.6.pt'
RESNET_WEIGHTS = 'weights/resnet18.pth'
TEST_IMAGE_DIR = 'datasetv3/test/images'
TEST_GT_DIR = 'datasetv3/test/labels'
DEVICE = 'cpu'
YOLO_THRESHOLD = 0.5 # Detections above this threshold are sent to hard gate

def main():
    print("--- Running Hybrid Model (Hard-Gate Method) ---")
    yolo_model = load_yolo_model(YOLO_WEIGHTS)
    resnet_model = load_resnet18(RESNET_WEIGHTS, device=DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    all_predictions = {}
    image_files = os.listdir(TEST_IMAGE_DIR)
    
    for img_file in tqdm(image_files, desc="Running Hard-Gate Pipeline"):
        img_path = os.path.join(TEST_IMAGE_DIR, img_file)
        all_predictions[img_file] = run_hard_gate_pipeline(
            yolo_model, resnet_model, img_path, transform, YOLO_THRESHOLD, DEVICE
        )
        
    evaluate_detections(all_predictions, TEST_GT_DIR, TEST_IMAGE_DIR, results_prefix="hybrid_hard_gate")

if __name__ == "__main__":
    main()