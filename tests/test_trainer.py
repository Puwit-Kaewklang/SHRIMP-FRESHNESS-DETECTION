import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest
from src.models import build_model
from src.trainer import train_model, EarlyStopping

def test_early_stopping():
    es = EarlyStopping(patience=2, min_delta=0.0)
    assert not es(1.0)
    assert not es(0.9)  # improvement
    assert not es(0.95) # no improvement (counter = 1)
    assert es(0.96)     # no improvement (counter = 2 >= patience -> True)

def test_train_model_single_epoch(tmp_path):
    # Create small dummy dataset
    X = torch.randn(8, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)

    model = build_model("custom_cnn", num_classes=2, pretrained=False)
    save_path = str(tmp_path / "best_model.pth")
    history = train_model(
        model,
        loader,
        loader,
        device=torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        save_path=save_path
    )

    assert "train_loss" in history
    assert "val_loss" in history
    assert "train_acc" in history
    assert "val_acc" in history
    assert len(history["val_loss"]) == 1
    assert os.path.exists(save_path)

def test_train_model_saves_history_json(tmp_path):
    X = torch.randn(8, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)

    model = build_model("custom_cnn", num_classes=2, pretrained=False)
    save_path = str(tmp_path / "best_model.pth")
    history_path = str(tmp_path / "history_custom_cnn.json")
    train_model(
        model,
        loader,
        loader,
        device=torch.device("cpu"),
        epochs=1,
        save_path=save_path,
        history_path=history_path
    )
    assert os.path.exists(history_path)

