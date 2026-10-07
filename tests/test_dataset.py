import os
import shutil
import zipfile
from PIL import Image
import pytest
from src.dataset import extract_and_prepare_dataset, get_data_loaders

@pytest.fixture
def dummy_zip(tmp_path):
    zip_dir = tmp_path / "raw"
    zip_dir.mkdir()
    fresh_dir = zip_dir / "fresh"
    not_fresh_dir = zip_dir / "not_fresh"
    fresh_dir.mkdir()
    not_fresh_dir.mkdir()
    
    # Create 10 dummy images per class (total 20)
    for i in range(10):
        img = Image.new("RGB", (32, 32), color=(i * 20, 100, 50))
        img.save(fresh_dir / f"fresh_{i}.jpg")
        img.save(not_fresh_dir / f"not_fresh_{i}.jpg")
        
    zip_file = tmp_path / "shrimp_raw_jpg.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        for root, _, files in os.walk(zip_dir):
            for file in files:
                abs_p = os.path.join(root, file)
                rel_p = os.path.relpath(abs_p, zip_dir)
                zf.write(abs_p, rel_p)
    return str(zip_file)

def test_extract_and_loaders(dummy_zip, tmp_path):
    extract_to = str(tmp_path / "extracted")
    data_dir = extract_and_prepare_dataset(dummy_zip, extract_to)
    assert os.path.exists(data_dir)
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        data_dir, batch_size=4, seed=42, split=(0.7, 0.15, 0.15)
    )
    assert set(classes) == {"fresh", "not_fresh"}
    assert len(train_loader.dataset) == 14  # 70% of 20
    assert len(val_loader.dataset) == 3    # 15% of 20
    assert len(test_loader.dataset) == 3   # 15% of 20
    
    # Check batch shapes from loader
    images, labels = next(iter(train_loader))
    assert images.shape[1:] == (3, 224, 224)
