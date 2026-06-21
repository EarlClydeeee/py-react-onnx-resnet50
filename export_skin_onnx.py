"""
Export the skin disease ResNet50 model to ONNX for browser inference.

Usage:
  python export_skin_onnx.py
  python export_skin_onnx.py --checkpoint path/to/best_resnet50_skin.pth

Without a checkpoint, exports ImageNet-pretrained backbone + untrained 7-class
head (app loads; predictions are not meaningful until you train in Colab and
re-run with --checkpoint).
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

NUM_CLASSES = 7
CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
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
    parser = argparse.ArgumentParser(description="Export skin ResNet50 to ONNX")
    parser.add_argument(
        "--checkpoint",
        help="Optional .pth from Colab training (best_resnet50_skin.pth)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("public", "models", "resnet50_skin.onnx"),
        help="Output ONNX path",
    )
    args = parser.parse_args()

    print("Building ResNet50 with 7-class skin disease head...")
    model = build_model()

    if args.checkpoint:
        print(f"Loading trained weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
    else:
        print(
            "No checkpoint provided — using ImageNet backbone + random fc head.\n"
            "Train in skin_classifier.ipynb, then re-run with:\n"
            "  python export_skin_onnx.py --checkpoint best_resnet50_skin.pth"
        )

    model.eval()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Exporting to {args.output}...")
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
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    labels_path = os.path.join("public", "data", "skin_labels.json")
    labels = [CLASS_NAMES[c] for c in CLASS_ORDER]
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
        f.write("\n")

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Done. Model: {args.output} ({size_mb:.1f} MB)")
    print(f"Labels:  {labels_path}")


if __name__ == "__main__":
    main()
