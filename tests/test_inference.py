import os
import torch
import numpy as np
from PIL import Image
import pytest
from src.inference import (
    get_eval_transform,
    find_model_weights,
    load_model,
    predict_image,
    generate_gradcam,
    CLASS_NAMES,
)

@pytest.fixture
def dummy_image():
    return Image.new("RGB", (300, 200), color=(100, 150, 200))

def test_get_eval_transform(dummy_image):
    transform = get_eval_transform()
    tensor = transform(dummy_image)
    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)

def test_find_model_weights():
    # Should find weights in models/5 models if present
    path = find_model_weights("resnet50")
    if os.path.exists("models/5 models/best_resnet50.pth"):
        assert path is not None
        assert os.path.exists(path)
    else:
        # If not present, returns None
        assert path is None or isinstance(path, str)

def test_load_model():
    device = torch.device("cpu")
    model = load_model("custom_cnn", weights_path=None, device=device)
    assert isinstance(model, torch.nn.Module)
    assert not model.training  # Must be in eval mode

def test_predict_image(dummy_image):
    device = torch.device("cpu")
    model = load_model("custom_cnn", weights_path=None, device=device)
    result = predict_image(model, dummy_image, device=device)
    
    assert "class_name" in result
    assert result["class_name"] in CLASS_NAMES
    assert "class_idx" in result
    assert result["class_idx"] in [0, 1]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 100.0
    assert "probabilities" in result
    assert "fresh" in result["probabilities"]
    assert "not_fresh" in result["probabilities"]
    assert pytest.approx(result["probabilities"]["fresh"] + result["probabilities"]["not_fresh"], rel=1e-3) == 100.0
    assert "latency_ms" in result
    assert result["latency_ms"] >= 0.0

def test_generate_gradcam_cnn(dummy_image):
    device = torch.device("cpu")
    model = load_model("custom_cnn", weights_path=None, device=device)
    cam_img = generate_gradcam(model, "custom_cnn", dummy_image, device=device)
    
    assert cam_img is not None
    assert isinstance(cam_img, np.ndarray)
    assert cam_img.ndim == 3
    assert cam_img.shape[2] == 3


def test_download_model_weights_if_missing_local_exists():
    from src.inference import download_model_weights_if_missing
    if os.path.exists("models/5 models/best_custom_cnn.pth"):
        path = download_model_weights_if_missing("custom_cnn")
        assert path is not None
        assert os.path.exists(path)


def test_download_model_weights_if_missing_remote_fallback():
    from src.inference import download_model_weights_if_missing
    path = download_model_weights_if_missing("non_existent_model_xyz", repo_id=None)
    assert path is None


def test_download_model_weights_mock_download(tmp_path):
    from unittest.mock import patch
    from src.inference import download_model_weights_if_missing

    with patch("src.inference.find_model_weights", return_value=None):
        with patch("huggingface_hub.hf_hub_download") as mock_hf:
            mock_file = tmp_path / "best_custom_cnn.pth"
            mock_file.write_text("dummy")
            mock_hf.return_value = str(mock_file)

            result = download_model_weights_if_missing("custom_cnn", repo_id="dummy/repo", save_dir=str(tmp_path))
            assert result == str(mock_file)
            mock_hf.assert_called_once()

