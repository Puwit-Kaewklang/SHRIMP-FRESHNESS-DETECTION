import os
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pytest

APP_PATH = str(Path(__file__).parent.parent / "app.py")

def test_app_initial_render():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    # Check that main title exists
    assert any("ระบบจำแนกความสดของกุ้งด้วย AI" in str(elem.value) for elem in at.markdown)

def test_app_model_selector():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    # Ensure selectbox for models is present
    assert len(at.sidebar.selectbox) >= 1
    # Check that ResNet-50 is the default
    assert at.sidebar.selectbox[0].value == "resnet50"


def test_app_file_uploader_rendered():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=15)
    assert not at.exception
    # Ensure file_uploader widget is active and rendered
    assert len(at.file_uploader) >= 1
    assert "file_uploader_" in at.file_uploader[0].key

