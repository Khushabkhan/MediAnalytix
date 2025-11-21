import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import gdown

# --- 1. CONFIGURATION & DOWNLOADER ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists('models'):
    os.makedirs('models')

PATH_DENSENET = "models/densenet121_focal.pth"
PATH_RESNET = "models/resnet50_weighted.pth"

# !!! PASTE YOUR GOOGLE DRIVE IDs HERE !!!
DENSENET_ID = "PASTE_YOUR_DENSENET_ID_HERE"
RESNET_ID   = "PASTE_YOUR_RESNET_ID_HERE"

def download_if_missing(file_path, file_id):
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, file_path, quiet=False)

# Download models on startup
try:
    download_if_missing(PATH_DENSENET, DENSENET_ID)
    download_if_missing(PATH_RESNET, RESNET_ID)
except Exception as e:
    print(f"Error downloading models: {e}")

# --- 2. LOAD MODELS ---
def load_model_architecture(model_type):
    if model_type == 'densenet':
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, 14)
    else:
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 14)
    return model

print("Loading models...")
model_dn = load_model_architecture('densenet')
model_rn = load_model_architecture('resnet')

try:
    model_dn.load_state_dict(torch.load(PATH_DENSENET, map_location=DEVICE))
    model_rn.load_state_dict(torch.load(PATH_RESNET, map_location=DEVICE))
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading weights: {e}")

model_dn = model_dn.to(DEVICE).eval()
model_rn = model_rn.to(DEVICE).eval()

# --- 3. TRANSFORMS ---
IMAGE_SIZE = 224
valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. GRAD-CAM SETUP ---
target_layers = [model_dn.features[-1]]
cam = GradCAM(model=model_dn, target_layers=target_layers)

# --- 5. PREDICTION ENGINE ---
def analyze_xray(image):
    if image is None: return None, "Please upload an image."
    
    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.float32(image_resized) / 255
    input_tensor = valid_transform(image).unsqueeze(0).to(DEVICE)
    
    # TTA
    image_flipped = cv2.flip(image, 1)
    tensor_flipped = valid_transform(image_flipped).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        dn_norm = F.sigmoid(model_dn(input_tensor))[0]
        dn_flip = F.sigmoid(model_dn(tensor_flipped))[0]
        prob_dn = (dn_norm + dn_flip) / 2

        rn_norm = F.sigmoid(model_rn(input_tensor))[0]
        rn_flip = F.sigmoid(model_rn(tensor_flipped))[0]
        prob_rn = (rn_norm + rn_flip) / 2
        
        final_prob = (prob_dn + prob_rn) / 2

    LABEL_COLUMNS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']
    
    all_scores = {label: prob.item() for label, prob in zip(LABEL_COLUMNS, final_prob)}
    sorted_findings = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    top_3 = sorted_findings[:3]
    primary_name = top_3[0][0]
    primary_score = top_3[0][1]
    
    valid_findings = []
    for name, score in top_3:
        if name == 'Enlarged Cardiomediastinum' and score < 0.60: continue
        if score > 0.10 and score >= (primary_score * 0.5):
            valid_findings.append((name, score))

    report = "# 🩺 Dual-Model Consensus Report\n\n"
    if not valid_findings:
        report += "### 🟢 No Significant Abnormalities Detected\n"
    else:
        p_name, p_score = valid_findings[0]
        icon = "🔴" if p_score > 0.5 else "🟠"
        report += f"### {icon} Primary Finding: **{p_name}**\n**Consensus Confidence:** {p_score*100:.1f}%\n\n"
        if len(valid_findings) > 1:
            report += "### ⚠️ Associated Findings\n"
            for name, score in valid_findings[1:]:
                report += f"- **{name}**: {score*100:.1f}%\n"

    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    except:
        visualization = image_resized

    return visualization, report

# --- 6. LAUNCH ---
if __name__ == "__main__":
    iface = gr.Interface(
        fn=analyze_xray,
        inputs=gr.Image(label="Upload Chest X-Ray"),
        outputs=[gr.Image(label="Ensemble Heatmap"), gr.Markdown(label="Report")],
        title="CheXpert AI Ensemble",
        allow_flagging="never"
    )
    iface.launch()
