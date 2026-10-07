import time
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics_from_arrays(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Computes standard classification metrics and confusion matrix."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_t, y_p, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])

    result: Dict[str, Any] = {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm,
    }

    if y_probs is not None:
        try:
            auc = float(roc_auc_score(y_t, y_probs))
            result["roc_auc"] = auc
        except Exception:
            result["roc_auc"] = None

    return result


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluates a model on a DataLoader, computing classification metrics
    and mean inference latency in milliseconds per image.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            start = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_samples += batch_size

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    metrics = compute_metrics_from_arrays(all_targets, all_preds, all_probs)
    latency_ms = (total_time / max(total_samples, 1)) * 1000.0

    metrics["latency_ms"] = float(latency_ms)
    metrics["y_true"] = all_targets
    metrics["y_pred"] = all_preds
    metrics["y_probs"] = all_probs

    return metrics


def create_leaderboard(
    eval_results: Dict[str, Dict[str, Any]],
    param_counts: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """Generates a sorted comparison DataFrame across evaluated models."""
    rows = []
    for name, res in eval_results.items():
        row = {
            "Model": name,
            "Accuracy": f"{res['accuracy'] * 100:.2f}%",
            "Precision": f"{res['precision'] * 100:.2f}%",
            "Recall": f"{res['recall'] * 100:.2f}%",
            "F1-Score": f"{res['f1_score'] * 100:.2f}%",
            "Latency (ms/img)": f"{res['latency_ms']:.2f} ms",
        }
        if param_counts and name in param_counts:
            row["Params (M)"] = f"{param_counts[name] / 1e6:.2f}M"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_confusion_matrices(
    cms_dict: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None
):
    """Plots 2x2 grid of confusion matrices for up to 4 models."""
    model_names = list(cms_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, name in enumerate(model_names[:4]):
        ax = axes[idx]
        cm = cms_dict[name]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    all_histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None
):
    """Plots Train and Validation Loss and Accuracy curves for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in all_histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax1.plot(epochs, hist["val_loss"], label=f"{name} (Val)")
        ax2.plot(epochs, hist["val_acc"], label=f"{name} (Val)")

    ax1.set_title("Validation Loss Comparison", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.set_title("Validation Accuracy Comparison", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
