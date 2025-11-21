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

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists('models'):
    os.makedirs('models')

PATH_DENSENET = "models/densenet121_focal.pth"
PATH_RESNET = "models/resnet50_weighted.pth"

DENSENET_ID = "https://drive.google.com/file/d/1Q1bYBIACogQLctmhJF7deE_qaO_nmGlj/view?usp=sharing"
RESNET_ID   = "https://drive.google.com/file/d/1H0DbZcSCG1uCSPytRLqX8VNUTq2QhiP3/view?usp=sharingE"

def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded!")

download_model(DENSENET_ID, PATH_DENSENET)
download_model(RESNET_ID, PATH_RESNET)

# Model
def load_model(path, model_type):
    if model_type == 'densenet':
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, 14)
    else: 
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 14)
    
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"✅ Loaded {model_type.upper()} model.")
    except Exception as e:
        print(f"❌ ERROR loading {model_type.upper()}: {e}")

    return model.to(DEVICE).eval()

model_dn = load_model(PATH_DENSENET, 'densenet')
model_rn = load_model(PATH_RESNET, 'resnet')

# Resize
IMAGE_SIZE = 224
valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad Cam
target_layers = [model_dn.features[-1]] # Use DenseNet for visual features
cam = GradCAM(model=model_dn, target_layers=target_layers)

#Ensemble prediction
def analyze_xray(image):
    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.float32(image_resized) / 255
    input_tensor = valid_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob_dn = F.sigmoid(model_dn(input_tensor))[0]
        prob_rn = F.sigmoid(model_rn(input_tensor))[0]
        final_prob = (prob_dn + prob_rn) / 2

    all_scores = {label: prob.item() for label, prob in zip(LABEL_COLUMNS, final_prob)}
    sorted_findings = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_findings[:3]
    
    high_conf = []
    probable = []
    suspicious = []
    
    for name, score in sorted_findings:
        percentage = score * 100
        
        if name == 'Enlarged Cardiomediastinum' and percentage < 60: continue
            
        if percentage > 60:
            high_conf.append(f"**{name}**: {percentage:.1f}%")
        elif percentage > 40:
            probable.append(f"{name}: {percentage:.1f}%")
        elif percentage > 15:
            suspicious.append(f"{name}: {percentage:.1f}%")

    # Final Report 
    top_name = sorted_findings[0][0]
    top_score = sorted_findings[0][1]

    report = f"### 🔎 Primary Finding: {top_name} ({top_score*100:.1f}%)\n\n"
    
    if high_conf: report += "### 🔴 High Confidence (>60%)\n" + "\n".join(high_conf) + "\n\n"
    if probable: report += "### 🟠 Probable (40-60%)\n" + "\n".join(probable) + "\n\n"
    if suspicious: report += "### 🟡 Suspicious (15-40%)\n" + "\n".join(suspicious) + "\n\n"
    if not (high_conf or probable or suspicious): report += "### 🟢 No Significant Abnormalities Detected\n"

    #Heatmap
    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    except Exception as e:
        visualization = image_resized 

    return visualization, report

    #UI
if __name__ == "__main__":
    iface = gr.Interface(
        fn=analyze_xray,
        inputs=gr.Image(label="Upload Chest X-Ray"),
        outputs=[
            gr.Image(label="Ensemble Heatmap Analysis"), 
            gr.Markdown(label="Consensus Report")
        ],
        title="CheXpert AI Assistant: Dual-Model Ensemble",
        description="Dual-Model System (Focal Loss DenseNet + Weighted ResNet) for high-accuracy diagnosis.",
        allow_flagging="never"
    )

    iface.launch(share=True)