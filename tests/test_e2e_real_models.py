from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from src.models import SUPPORTED_MODELS
from src.inference import (
    load_model,
    predict_image,
    generate_gradcam,
    find_model_weights,
    CLASS_NAMES,
)

def get_current_sample_images():
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for cat in ["fresh", "not_fresh"]:
        p = Path(f"sample_images/{cat}")
        if p.exists():
            for f in p.iterdir():
                if f.suffix.lower() in valid_exts and not f.name.startswith("."):
                    paths.append(str(f))
    return paths

@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_real_model_weights_and_sample_inference(model_name):
    weights_path = find_model_weights(model_name)
    assert weights_path is not None, f"Weights for {model_name} should exist in models"
    
    device = torch.device("cpu")
    model = load_model(model_name, weights_path=weights_path, device=device)
    assert isinstance(model, torch.nn.Module)
    
    sample_paths = get_current_sample_images()
    assert len(sample_paths) > 0, "At least one sample image should exist"
    
    for img_path in sample_paths:
        img = Image.open(img_path)
        result = predict_image(model, img, device=device)
        assert result["class_name"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 100.0
        assert "latency_ms" in result
        assert result["latency_ms"] > 0.0

@pytest.mark.parametrize("cnn_model_name", ["custom_cnn", "mobilenet_v3", "resnet50", "efficientnet_b0"])
def test_real_cnn_gradcam_generation(cnn_model_name):
    weights_path = find_model_weights(cnn_model_name)
    device = torch.device("cpu")
    model = load_model(cnn_model_name, weights_path=weights_path, device=device)
    
    sample_paths = get_current_sample_images()
    assert len(sample_paths) > 0
    test_img = Image.open(sample_paths[0])
    cam = generate_gradcam(model, cnn_model_name, test_img, device=device)
    
    assert cam is not None
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224, 3)

def test_vit_gradcam_handling():
    weights_path = find_model_weights("vit_b_16")
    device = torch.device("cpu")
    model = load_model("vit_b_16", weights_path=weights_path, device=device)
    
    sample_paths = get_current_sample_images()
    assert len(sample_paths) > 0
    test_img = Image.open(sample_paths[0])
    cam = generate_gradcam(model, "vit_b_16", test_img, device=device)
    # ViT has no standard Conv2d layer; should gracefully return None
    assert cam is None
