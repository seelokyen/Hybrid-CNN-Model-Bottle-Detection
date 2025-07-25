import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from torchvision.models import resnet18

# Model Loading Functions
def load_yolo_model(weights_path):
    # Loads a YOLOv8n pre-trained model
    return YOLO(weights_path)

def load_resnet18(weights_path, device='cpu'):
    # Loads a pre-trained ResNet18 model with a modified final layer
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- Inference Pipeline Functions ---
def run_yolo_inference(model, image_path, conf_thres):
    # Runs YOLOv8 inference and returns raw detections
    detections = []
    result = model(image_path, verbose=False)[0]
    for box in result.boxes:
        conf = float(box.conf.item())
        if conf < conf_thres:
            continue
        cls = int(box.cls.item())
        if cls != 0: # Assuming 'bottle' is class 0
            continue
        detections.append({
            "bbox": list(map(int, box.xyxy[0].tolist())),
            "conf": conf,
        })
    return detections

def run_hard_gate_pipeline(yolo_model, resnet_model, image_path, transform, yolo_thresh, device):
    """Runs the Yes/No hard-gating hybrid pipeline."""
    final_detections = []
    pil_img = Image.open(image_path).convert('RGB')
    
    # Get initial detections, trusting those above the threshold
    yolo_dets = run_yolo_inference(yolo_model, image_path, conf_thres=0.01) # Use a low threshold to get all candidates

    for det in yolo_dets:
        if det['conf'] >= yolo_thresh:
            det['pred'] = 1
            final_detections.append(det)
        else: # Ambiguous case, send to ResNet
            crop = pil_img.crop(det['bbox'])
            if crop.size[0] > 0 and crop.size[1] > 0:
                with torch.no_grad():
                    tensor = transform(crop).unsqueeze(0).to(device)
                    output = resnet_model(tensor)
                    pred = torch.argmax(output, dim=1).item()
                    if pred == 1:
                        det['pred'] = 1
                        final_detections.append(det)
    return final_detections
    
def run_ensemble_pipeline(yolo_model, resnet_model, image_path, transform, weights, high_conf_thresh, final_thresh, device):
    """
    Runs the weighted average ensemble pipeline using a simplified two-tiered logic.
    """
    final_detections = []
    pil_img = Image.open(image_path).convert('RGB')
    
    # We still need to get all potential candidates, so we use a very low threshold here.
    # The actual filtering happens inside the loop.
    yolo_dets = run_yolo_inference(yolo_model, image_path, conf_thres=0.01)

    for det in yolo_dets:
        yolo_score = det['conf']
        bbox = det['bbox']

        # --- Simplified Two-Tiered Logic ---
        if yolo_score >= high_conf_thresh:
            # High confidence, accept it directly
            det['pred'] = 1
            final_detections.append(det)
            continue
        
        # --- If not high confidence, it's an ambiguous case ---
        crop = pil_img.crop(bbox)
        if crop.size[0] == 0 or crop.size[1] == 0:
            continue
        
        with torch.no_grad():
            tensor = transform(crop).unsqueeze(0).to(device)
            logits = resnet_model(tensor)
            resnet_prob = F.softmax(logits, dim=1)[0, 1].item()
        
        final_score = (weights['yolo'] * yolo_score) + (weights['resnet'] * resnet_prob)
        
        if final_score >= final_thresh:
            det['pred'] = 1
            det['conf'] = final_score
            final_detections.append(det)
            
    return final_detections