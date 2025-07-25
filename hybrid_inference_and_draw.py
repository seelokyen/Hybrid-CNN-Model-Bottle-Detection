import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from src.models import load_resnet18

# === Configuration ===
image_dir = 'dataset/test/images'
output_dir = 'results/annotated'
yolo_weights = 'weights/yolov8n.pt'
resnet_weights = 'weights/resnet18.pth'
conf_thresh = 0.5

os.makedirs(output_dir, exist_ok=True)

# === Load models ===
print("Loading YOLOv8 model...")
yolo_model = YOLO(yolo_weights)

print("Loading ResNet18 classifier...")
resnet_model = load_resnet18(resnet_weights)

# === Define ResNet preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === Process each image ===
for img_file in os.listdir(image_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read image: {img_file}")
        continue
    orig = img.copy()

    print(f"📷 Processing: {img_file}")
    result = yolo_model(img_path)[0]

    for box in result.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Skip if not bottle class
        if cls != 0:
            continue

        label = ""
        color = (0, 255, 0)  # green → confident YOLO

        if conf >= conf_thresh:
            label = f"bottle {conf:.2f}"
        else:
            # Send to ResNet18
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_crop).unsqueeze(0)

            with torch.no_grad():
                out = resnet_model(input_tensor)
                pred = torch.argmax(out, dim=1).item()

            if pred == 1:
                label = "bottle"
                color = (255, 0, 0)  # blue
            else:
                label = "not_bottle"
                color = (0, 0, 255)  # red

        # Draw bounding box & label
        cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
        cv2.putText(orig, label, (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save annotated image
    save_path = os.path.join(output_dir, img_file)
    cv2.imwrite(save_path, orig)
    print(f"Saved: {save_path}")

print("\nAnnotated images saved to:", output_dir)
