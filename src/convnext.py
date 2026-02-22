import torch
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load ConvNeXt Model Architecture ----
convnext_model = models.convnext_tiny(weights=None)   # No pretrained weights
convnext_model.classifier[2] = torch.nn.Linear(768, 2)  # Modify to match your dataset (2 classes)

# ---- Load Saved Weights ----
state_dict = torch.load("./models/best_convnext_model.pth", map_location=device)
convnext_model.load_state_dict(state_dict)

convnext_model.to(device)
convnext_model.eval()

print("ConvNeXt model loaded successfully (weights-only)! 🚀")


# ---- Preprocessing ----
inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_map = {0: "Real", 1: "AI-Generated (Fake)"}


def convnext_predict(img_bytes):
    """Runs inference and returns prediction + confidence."""
    
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = convnext_model(img_tensor)
        probabilities = F.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

    return (
        label_map[int(predicted.item())], # prediction
        round(float(confidence.item()) * 100, 2) # confidence
    )
