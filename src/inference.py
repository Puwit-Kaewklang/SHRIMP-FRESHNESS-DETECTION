from typing import Dict, List, Optional, Tuple, Any
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.cm as cm

from src.models import build_model, SUPPORTED_MODELS

CLASS_NAMES: List[str] = ["fresh", "not_fresh"]

MODEL_FILENAMES: Dict[str, str] = {
    "custom_cnn": "best_custom_cnn.pth",
    "mobilenet_v3": "best_mobilenet_v3.pth",
    "resnet50": "best_resnet50.pth",
    "efficientnet_b0": "best_efficientnet_b0.pth",
    "vit_b_16": "best_vit_b_16.pth",
}

DEFAULT_SEARCH_DIRS: List[str] = [
    "models",
    "models/5 models",
    "../models",
    "../models/5 models",
]


def find_model_weights(model_name: str, search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """
    Search for model weights (.pth) across designated model directories.
    """
    dirs = search_dirs or DEFAULT_SEARCH_DIRS
    target_filename = MODEL_FILENAMES.get(model_name, f"best_{model_name}.pth")

    for directory in dirs:
        if not os.path.exists(directory):
            continue
        candidate = os.path.join(directory, target_filename)
        if os.path.isfile(candidate):
            return candidate

        # Fuzzy check inside directory
        for f in os.listdir(directory):
            if f.endswith(".pth") and model_name.replace("_", "").lower() in f.replace("_", "").lower():
                return os.path.join(directory, f)

    return None


def get_eval_transform() -> transforms.Compose:
    """
    Standard evaluation transform: Resize to 256, CenterCrop to 224, ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(
    model_name: str,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Build model architecture and optionally load trained weights.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)

    resolved_path = weights_path or find_model_weights(model_name)
    if resolved_path and os.path.isfile(resolved_path):
        state_dict = torch.load(resolved_path, map_location=dev)
        model.load_state_dict(state_dict)

    model.to(dev)
    model.eval()
    return model


def predict_image(
    model: nn.Module,
    image: Image.Image,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run inference on a single PIL image and return predictions, probabilities, and latency.
    """
    dev = device or next(model.parameters()).device
    transform = get_eval_transform()

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(dev)

    # Measure latency
    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    class_idx = int(torch.argmax(probs).item())
    class_name = CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx].item() * 100.0)

    probabilities = {
        CLASS_NAMES[i]: float(probs[i].item() * 100.0)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "class_idx": class_idx,
        "class_name": class_name,
        "confidence": confidence,
        "probabilities": probabilities,
        "latency_ms": latency_ms,
        "logits": logits.cpu().numpy().tolist(),
    }


def get_target_layer_for_gradcam(model: nn.Module, model_name: str) -> Optional[nn.Module]:
    """
    Return the final convolutional layer for Grad-CAM extraction based on architecture.
    """
    if model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "mobilenet_v3":
        return model.features[-1]
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    elif model_name == "custom_cnn":
        # Block 4 Conv2d is at index 12 in features sequential
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                return layer
        return model.features[-2]
    elif model_name == "vit_b_16":
        # ViT has no standard Conv2d layer
        return None
    return None


def generate_gradcam(
    model: nn.Module,
    model_name: str,
    image: Image.Image,
    device: Optional[torch.device] = None,
    target_class: Optional[int] = None,
    alpha: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM activation heatmap overlaid on the input image.
    Returns RGB uint8 numpy array of shape (H, W, 3), or None if unsupported (e.g. ViT).
    """
    target_layer = get_target_layer_for_gradcam(model, model_name)
    if target_layer is None:
        return None

    dev = device or next(model.parameters()).device
    transform = get_eval_transform()

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(dev)
    tensor.requires_grad = True

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(tensor)

    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    score = logits[0, target_class]
    score.backward()

    f_handle.remove()
    b_handle.remove()

    if not activations or not gradients:
        return None

    act = activations[0].detach()  # Shape: (1, C, H, W)
    grad = gradients[0].detach()   # Shape: (1, C, H, W)

    # Global average pooling over spatial dimensions for weights
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = torch.sum(weights * act, dim=1, keepdim=True)   # (1, 1, H, W)
    cam = F.relu(cam)

    cam_np = cam.squeeze().cpu().numpy()
    if np.max(cam_np) > 0:
        cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np))
    else:
        cam_np = np.zeros_like(cam_np)

    # Resize CAM to match 224x224 (the cropped image dimension)
    cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
    cam_resized = np.array(cam_pil) / 255.0

    # Apply colormap JET
    import matplotlib.pyplot as plt
    colormap = plt.get_cmap("jet")
    heatmap_colored = colormap(cam_resized)[:, :, :3]  # (224, 224, 3) in [0, 1]

    # Crop original image similarly to eval transform (Resize 256, CenterCrop 224)
    orig_resized = image.resize((256, int(256 * image.height / image.width))) if image.width > image.height else image.resize((int(256 * image.width / image.height), 256))
    w, h = orig_resized.size
    left = (w - 224) // 2
    top = (h - 224) // 2
    orig_cropped = orig_resized.crop((left, top, left + 224, top + 224))
    orig_np = np.array(orig_cropped) / 255.0

    # Blend original and heatmap
    overlay = (1.0 - alpha) * orig_np + alpha * heatmap_colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    return overlay
