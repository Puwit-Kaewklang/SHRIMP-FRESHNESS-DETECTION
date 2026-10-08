import os
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

SAMPLE_IMAGE_PATHS = [
    "sample_images/fresh/timthumb.jpeg",
    "sample_images/fresh/5.fresh-shrimp-prawn-1024x781.webp",
    "sample_images/not_fresh/images.jpeg",
    "sample_images/not_fresh/images (1).jpeg",
]

@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_real_model_weights_and_sample_inference(model_name):
    weights_path = find_model_weights(model_name)
    assert weights_path is not None, f"Weights for {model_name} should exist in models/5 models"
    assert os.path.exists(weights_path)
    
    device = torch.device("cpu")
    model = load_model(model_name, weights_path=weights_path, device=device)
    assert isinstance(model, torch.nn.Module)
    
    for img_path in SAMPLE_IMAGE_PATHS:
        assert os.path.exists(img_path), f"Sample image {img_path} not found"
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
    
    test_img = Image.open("sample_images/fresh/timthumb.jpeg")
    cam = generate_gradcam(model, cnn_model_name, test_img, device=device)
    
    assert cam is not None
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224, 3)

def test_vit_gradcam_handling():
    weights_path = find_model_weights("vit_b_16")
    device = torch.device("cpu")
    model = load_model("vit_b_16", weights_path=weights_path, device=device)
    
    test_img = Image.open("sample_images/fresh/timthumb.jpeg")
    cam = generate_gradcam(model, "vit_b_16", test_img, device=device)
    # ViT has no standard Conv2d layer; should gracefully return None
    assert cam is None
