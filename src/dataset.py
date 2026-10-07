import os
import zipfile
import shutil
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class TransformedSubset(Dataset):
    """Subset of a dataset with an applied transform."""
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def extract_and_prepare_dataset(source_path: str, extract_to: str) -> str:
    """
    Extracts zip archive or copies directory and identifies the root directory containing
    the 'fresh' and 'not_fresh' class folders.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Dataset path not found: {source_path}")

    os.makedirs(extract_to, exist_ok=True)
    if os.path.isdir(source_path):
        # If source is already a directory
        if os.path.abspath(source_path) != os.path.abspath(extract_to):
            # If fresh and not_fresh already directly inside source_path
            f_check = os.path.join(source_path, "fresh")
            nf_check = os.path.join(source_path, "not_fresh")
            if os.path.isdir(f_check) and os.path.isdir(nf_check):
                return source_path
            for item in os.listdir(source_path):
                s_item = os.path.join(source_path, item)
                d_item = os.path.join(extract_to, item)
                if os.path.isdir(s_item):
                    if os.path.exists(d_item):
                        shutil.rmtree(d_item)
                    shutil.copytree(s_item, d_item)
                else:
                    shutil.copy2(s_item, d_item)
    else:
        with zipfile.ZipFile(source_path, "r") as zf:
            zf.extractall(extract_to)

    # Search for folder containing fresh and not_fresh
    for root, dirs, _ in os.walk(extract_to):
        dir_set = set(d.lower() for d in dirs)
        if "fresh" in dir_set and "not_fresh" in dir_set:
            return root

    # If already directly inside extract_to
    fresh_p = os.path.join(extract_to, "fresh")
    not_fresh_p = os.path.join(extract_to, "not_fresh")
    if os.path.isdir(fresh_p) and os.path.isdir(not_fresh_p):
        return extract_to

    raise ValueError(f"Could not locate 'fresh' and 'not_fresh' folders inside {extract_to}")


def collect_samples(data_dir: str) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Scans fresh and not_fresh folders, filtering valid images."""
    class_names = ["fresh", "not_fresh"]
    samples: List[Tuple[str, int]] = []

    for label_idx, cname in enumerate(class_names):
        folder = os.path.join(data_dir, cname)
        if not os.path.isdir(folder):
            # Try matching case-insensitively
            for item in os.listdir(data_dir):
                if item.lower() == cname.lower():
                    folder = os.path.join(data_dir, item)
                    break

        for root, _, files in os.walk(folder):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in VALID_EXTENSIONS:
                    samples.append((os.path.join(root, file), label_idx))

    if not samples:
        raise ValueError(f"No valid images found in {data_dir}")

    return samples, class_names


def get_transforms():
    """Returns train and evaluation torchvision transforms."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    seed: int = 42,
    split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_workers: int = 0
):
    """
    Creates stratified Train, Validation, and Test PyTorch DataLoaders.
    """
    samples, class_names = collect_samples(data_dir)
    labels = [s[1] for s in samples]

    train_ratio, val_ratio, test_ratio = split
    assert abs(sum(split) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Split train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_samples, temp_samples, _, temp_labels = train_test_split(
        samples, labels, test_size=temp_ratio, random_state=seed, stratify=labels
    )

    # Split temp into val and test
    relative_test_ratio = test_ratio / temp_ratio
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=relative_test_ratio, random_state=seed, stratify=temp_labels
    )

    train_transform, eval_transform = get_transforms()

    train_dataset = TransformedSubset(train_samples, transform=train_transform)
    val_dataset = TransformedSubset(val_samples, transform=eval_transform)
    test_dataset = TransformedSubset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_names
