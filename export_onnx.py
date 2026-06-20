import os
import torch
import torchvision.models as models

def main():
    print("Loading pre-trained ResNet50 model from torchvision...")
    # Load model with default weights
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()

    # Create dummy input of shape (batch_size, channels, height, width)
    # ResNet50 expects 3-channel 224x224 images
    dummy_input = torch.randn(1, 3, 224, 224)

    # Define output directory
    output_dir = os.path.join("public", "models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "resnet50.onnx")

    print(f"Exporting model to ONNX format at: {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,        # Store trained parameter weights inside the model file
        opset_version=14,          # Opset 14 is widely supported by ONNX Runtime Web
        do_constant_folding=True,  # Constant folding optimization
        input_names=["input"],     # Define input name
        output_names=["output"],   # Define output name
        dynamic_axes={
            "input": {0: "batch_size"}, # Allow dynamic batch size
            "output": {0: "batch_size"}
        }
    )
    print("Model successfully exported to ONNX!")

if __name__ == "__main__":
    main()
