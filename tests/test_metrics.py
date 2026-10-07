import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest
from src.models import build_model
from src.metrics import evaluate_model, compute_metrics_from_arrays

def test_evaluate_model():
    X = torch.randn(8, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)

    model = build_model("custom_cnn", num_classes=2, pretrained=False)
    results = evaluate_model(model, loader, device=torch.device("cpu"))

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert "latency_ms" in results
    assert results["confusion_matrix"].shape == (2, 2)
    assert len(results["y_true"]) == 8
    assert len(results["y_pred"]) == 8

def test_compute_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = compute_metrics_from_arrays(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"].shape == (2, 2)
