import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import gradio as gr
from PIL import Image
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------
# Setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
NUM_CLASSES = len(CLASS_NAMES)

MODEL_INFO = {
    "ResNet50 (98.43% accuracy)": {
        "key": "resnet50",
        "accuracy": "98.43%",
        "weights_file": "pcb_resnet50.pth",
    },
    "MobileNetV2 (96.86% accuracy)": {
        "key": "mobilenetv2",
        "accuracy": "96.86%",
        "weights_file": "mobilenetv2_pcb_defects.pth",
    },
    "Custom CNN (88.12% accuracy)": {
        "key": "customcnn",
        "accuracy": "88.12%",
        "weights_file": "custom_cnn_pcb_defects.pth",
    },
}

DEFAULT_MODEL_LABEL = "ResNet50 (98.43% accuracy)"

# -----------------------------
# Custom CNN architecture (must match training definition exactly)
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -----------------------------
# Model builders + Grad-CAM target layers
# -----------------------------
def build_resnet50():
    m = models.resnet50(weights=None)
    num_features = m.fc.in_features
    m.fc = nn.Linear(num_features, NUM_CLASSES)
    target_layer = m.layer4[-1]
    return m, target_layer


def build_mobilenetv2():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(m.last_channel, NUM_CLASSES)
    )
    target_layer = m.features[-1]
    return m, target_layer


def build_customcnn():
    m = SimpleCNN(num_classes=NUM_CLASSES)
    target_layer = m.conv4
    return m, target_layer


BUILDERS = {
    "resnet50": build_resnet50,
    "mobilenetv2": build_mobilenetv2,
    "customcnn": build_customcnn,
}

# -----------------------------
# Lazy-load + cache models and their Grad-CAM objects
# -----------------------------
_loaded_models = {}
_loaded_cams = {}

def get_model_and_cam(label):
    info = MODEL_INFO[label]
    key = info["key"]

    if key not in _loaded_models:
        model, target_layer = BUILDERS[key]()
        state_dict = torch.load(info["weights_file"], map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _loaded_models[key] = model
        _loaded_cams[key] = GradCAM(model=model, target_layers=[target_layer])

    return _loaded_models[key], _loaded_cams[key]


# -----------------------------
# Preprocessing (must match training eval_transform)
# -----------------------------
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Prediction function
# -----------------------------
def predict_defect(input_image, model_label):
    if input_image is None:
        return None, None

    model, cam = get_model_and_cam(model_label)

    img_tensor = eval_transform(input_image.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()

    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    img_array = img_tensor[0].cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = std * img_array + mean
    img_array = np.clip(img_array, 0, 1)

    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    return confidences, visualization


# -----------------------------
# Styling
# -----------------------------
custom_css = """
.gradio-container {
    background: radial-gradient(circle at 20% 20%, #0d1b2a 0%, #060c12 60%, #02050a 100%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

#title_block h1 {
    color: #39ff8f !important;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-shadow: 0 0 18px rgba(57, 255, 143, 0.55), 0 0 2px rgba(57, 255, 143, 0.9);
}

.gr-block, .gr-box, .block {
    background: #0b1620 !important;
    border: 1px solid rgba(57, 255, 143, 0.25) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 25px rgba(0, 0, 0, 0.5), inset 0 0 0 1px rgba(57,255,143,0.05);
}

.tabs > .tab-nav button {
    color: #8fb6a8 !important;
    font-weight: 600;
}
.tabs > .tab-nav button.selected {
    color: #39ff8f !important;
    border-bottom: 2px solid #39ff8f !important;
    text-shadow: 0 0 8px rgba(57,255,143,0.6);
}

.gr-button-primary, button.primary {
    background: linear-gradient(90deg, #1fae63, #39ff8f) !important;
    color: #02120a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 0 20px rgba(57, 255, 143, 0.45) !important;
    transition: all 0.2s ease-in-out;
}
.gr-button-primary:hover, button.primary:hover {
    box-shadow: 0 0 30px rgba(57, 255, 143, 0.75) !important;
    transform: translateY(-1px);
}

.label-wrap, .output-class {
    background: #0b1620 !important;
    border-radius: 12px !important;
}

label, .gr-markdown, p, span {
    color: #cfe8dc !important;
}

.gr-markdown h3 {
    color: #39ff8f !important;
}

hr {
    border-color: rgba(57,255,143,0.2) !important;
}
"""

theme = gr.themes.Base(
    primary_hue="green",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#02050a",
    block_background_fill="#0b1620",
    block_border_color="#1fae63",
    border_color_primary="#39ff8f",
    block_radius="16px",
    button_primary_background_fill="#39ff8f",
    button_primary_text_color="#02120a",
)

# -----------------------------
# Interface
# -----------------------------
with gr.Blocks(theme=theme, css=custom_css, title="PCB Defect Classifier") as demo:
    gr.Markdown(
        """
        # Explainable PCB Defect Classifier
        """,
        elem_id="title_block"
    )

    with gr.Tabs():
        with gr.TabItem("Live Classification (Grad-CAM)"):
            gr.Markdown(
                "Upload a cropped PCB defect region, choose a model, and get a real-time prediction "
                "with a Grad-CAM explanation of which pixels drove the decision."
            )
            with gr.Row():
                with gr.Column():
                    model_selector = gr.Radio(
                        choices=list(MODEL_INFO.keys()),
                        value=DEFAULT_MODEL_LABEL,
                        label="Select Model"
                    )
                    input_img = gr.Image(type="pil", label="Upload PCB Defect Crop")
                    submit_btn = gr.Button("Analyze Defect", variant="primary")
                    gr.Examples(
                        examples=[
                            "Missing_hole_sample.jpg",
                            "Mouse_bite_sample.jpg",
                            "Open_circuit_sample.jpg",
                            "Short_sample.jpg",
                            "Spur_sample.jpg",
                            "Spurious_copper_sample.jpg",
                        ],
                        inputs=input_img,
                        label="Try a sample defect"
                    )
                with gr.Column():
                    output_label = gr.Label(num_top_classes=6, label="Prediction Confidence")
                    output_heatmap = gr.Image(label="Grad-CAM Explanation")

            submit_btn.click(
                fn=predict_defect,
                inputs=[input_img, model_selector],
                outputs=[output_label, output_heatmap]
            )

        with gr.TabItem("Offline Analysis (SHAP)"):
            gr.Markdown(
                """
                ### SHAP Class-Wise Attribution Analysis (ResNet50)
                SHAP provides deeper class-by-class attribution but requires hundreds of model evaluations per image,
                making it impractical for real-time use. Below are precomputed SHAP results for the lead model
                (ResNet50), showing which regions push predictions toward or away from each class.
                """
            )
            gr.Image(value="shap_results.png", label="SHAP Attribution Map (Precomputed, ResNet50)")

        with gr.TabItem("Model Comparison"):
            gr.Markdown(
                """
                ### Model Performance Summary

                | Model | Test Accuracy | Macro F1 | Total Params | Trainable Params |
                |---|---|---|---|---|
                | Custom CNN (Baseline) | 88.12% | 0.88 | 26.1M | 26.1M |
                | MobileNetV2 (Transfer Learning) | 96.86% | 0.97 | 2.23M | 1.21M |
                | ResNet50 (Transfer Learning) | 98.43% | 0.98 | 23.52M | ~23.5M |

                Both transfer-learning models substantially outperform the from-scratch baseline.
                MobileNetV2 achieves near-top accuracy with a fraction of the parameters, making it
                an attractive option for lightweight or edge deployment.
                """
            )

    gr.Markdown(
        """
        ---
        **Defect Classes:** Missing Hole &nbsp;|&nbsp; Mouse Bite &nbsp;|&nbsp; Open Circuit &nbsp;|&nbsp; Short &nbsp;|&nbsp; Spur &nbsp;|&nbsp; Spurious Copper
        """
    )

if __name__ == "__main__":
    demo.launch()
