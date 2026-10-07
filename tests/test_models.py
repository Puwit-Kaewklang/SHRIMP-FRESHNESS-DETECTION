import torch
import pytest
from src.models import build_model, SUPPORTED_MODELS, CustomCNN

@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_model_forward_pass(model_name):
    # Test with pretrained=False for fast offline test
    model = build_model(model_name, num_classes=2, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2)

def test_custom_cnn_structure():
    model = CustomCNN(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2)
