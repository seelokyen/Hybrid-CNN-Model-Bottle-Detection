# Hybrid CNN-YOLOv8 Bottle Detection

This project implements a hybrid detection system combining YOLOv8 and a ResNet18 classifier for robust bottle detection. It includes both ensemble and hard-gating strategies.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Hybrid-CNN-Model-Bottle-Detection.git
cd Hybrid-CNN-Model-Bottle-Detection
```

### 2. Create and activate a virtual environment
On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Required Packages
If requirements.txt is missing, install manually:
```bash
pip install torch torchvision ultralytics matplotlib scikit-learn tqdm opencv-python pillow
```

## Dataset Download (via Roboflow)
This project does not include the dataset due to file size and filename length issues.
To get the dataset:
1. Visit your Roboflow project:
   - Download Dataset on Roboflow URL: https://app.roboflow.com/lokyen/floating-bottle-detection-ib8tj/4
2. Choose YOLOv8 format when exporting
3. Unzip it and place the contents in:
```bash
dataset/
```
4. Your folder should look like:
```bash
dataset/
  train/
    images/
      image1.jpg
    labels/
      image1.txt
  valid/
    images/
      image1.jpg
    labels/
      image1.txt
  test/
    images/
      image1.jpg
    labels/
      image1.txt
  data.yaml
  README.dataset.txt
  README.roboflow.txt
```

## Scripts
| Script                     | Purpose                                           |
|---------------------------|---------------------------------------------------|
| `evaluate_baseline.py`    | Run YOLOv8-only evaluation                        |
| `run_hard_gate_hybrid.py` | Run hybrid detection using hard-gating            |
| `run_ensemble_hybrid.py`  | Run hybrid detection using weighted ensemble      |
| `hybrid_inference.py`     | Run inference and draw annotated output images (hard gate model)    |

## Evaluation Outputs
After running the hybrid scripts, evaluation metrics, confusion matrix, and PR curves will be saved in the ```results/``` folder.
