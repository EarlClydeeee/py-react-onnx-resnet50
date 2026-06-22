"""
Export the brain tumor ResNet50 model to ONNX for browser inference.

Usage (placeholder — app loads but predictions are not trained):
  python export_brain_onnx.py

Usage (after Colab training):
  python export_brain_onnx.py --checkpoint best_resnet50_brain.pth
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

NUM_CLASSES = 4
CLASS_ORDER = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_NAMES = {
    "glioma":     "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor":    "No Tumor",
    "pituitary":  "Pituitary Tumor",
}


def build_model():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CLASSES),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Export brain tumor ResNet50 to ONNX")
    parser.add_argument(
        "--checkpoint",
        help="Path to best_resnet50_brain.pth from Colab training",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("public", "models", "resnet50_brain.onnx"),
    )
    args = parser.parse_args()

    print("Building ResNet50 with 4-class brain tumor head...")
    model = build_model()

    if args.checkpoint:
        print(f"Loading trained weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
        print("Trained weights loaded.")
    else:
        print(
            "No checkpoint provided — using ImageNet backbone + random fc head.\n"
            "The app will load but predictions are not meaningful.\n"
            "Train in brain_tumor_classifier.ipynb, then re-run:\n"
            "  python export_brain_onnx.py --checkpoint best_resnet50_brain.pth"
        )

    model.eval()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting ONNX to {args.output}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    labels_path = os.path.join("public", "data", "brain_labels.json")
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    labels = [CLASS_NAMES[c] for c in CLASS_ORDER]
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
        f.write("\n")

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Done.")
    print(f"  Model:  {args.output} ({size_mb:.1f} MB)")
    print(f"  Labels: {labels_path}")


if __name__ == "__main__":
    main()
