#!/usr/bin/env python3
"""
Upload Trained Model Weights to Hugging Face Model Hub (Free Storage)
-------------------------------------------------------------------
This utility uploads the 5 trained model weights (.pth) to a Hugging Face
Model Repository so that Streamlit Community Cloud can download them on-demand
without violating GitHub's 100MB file limit.

Prerequisites:
    1. Create a free account at https://huggingface.co/
    2. Create a Write Access Token at https://huggingface.co/settings/tokens
    3. Run:
       pip install huggingface_hub
       python scripts/upload_models_to_hf.py --repo <username/shrimp-freshness-models> --token <hf_token>
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

MODEL_FILES = [
    "best_custom_cnn.pth",
    "best_mobilenet_v3.pth",
    "best_resnet50.pth",
    "best_efficientnet_b0.pth",
    "best_vit_b_16.pth",
]

DEFAULT_SEARCH_DIRS = [
    Path("models/5 models"),
    Path("models"),
]


def find_local_weights() -> dict:
    found = {}
    for filename in MODEL_FILES:
        for sdir in DEFAULT_SEARCH_DIRS:
            candidate = sdir / filename
            if candidate.is_file():
                found[filename] = candidate
                break
    return found


def main():
    parser = argparse.ArgumentParser(description="Upload shrimp freshness models to Hugging Face Hub")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face repo id, e.g. username/shrimp-models")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (Write permission)")
    parser.add_argument("--private", action="store_true", help="Set repository to private (default is public)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"📦 Checking local model weights in 'models/'...")
    found_weights = find_local_weights()

    if not found_weights:
        print("❌ Error: No .pth model weights found in 'models/5 models/' or 'models/'.")
        sys.exit(1)

    print(f"Found {len(found_weights)} / {len(MODEL_FILES)} model weight files:")
    for fname, fpath in found_weights.items():
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  ✓ {fname} ({size_mb:.1f} MB) -> {fpath}")

    # Create repo if not exists
    print(f"\n🚀 Verifying/Creating Hugging Face Model Repository: '{args.repo}'...")
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="model",
            token=token,
            private=args.private,
            exist_ok=True,
        )
        print(f"✓ Repository ready: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"⚠️ Note during repo creation: {e}")

    # Upload files
    print("\n📤 Uploading model weights to Hugging Face...")
    for filename, local_path in found_weights.items():
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  Uploading {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=args.repo,
                repo_type="model",
                token=token,
            )
            print("Done! ✅")
        except Exception as e:
            print(f"Failed! ❌\n  Error: {e}")

    print("\n🎉 Upload process completed!")
    print(f"Your models are hosted at: https://huggingface.co/{args.repo}")
    print(f"\nTo configure Streamlit Community Cloud, set this Secret in Streamlit App Settings:")
    print(f'HF_MODELS_REPO = "{args.repo}"')


if __name__ == "__main__":
    main()
