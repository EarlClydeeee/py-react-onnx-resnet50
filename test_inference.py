import os
import shutil
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image_path):
    print(f"Loading and preprocessing image from: {image_path}...")
    # Load and convert image to RGB
    image = Image.open(image_path).convert("RGB")
    
    # Define standard ImageNet transformation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = preprocess(image)
    batch_tensor = tensor.unsqueeze(0) # Add batch dimension -> (1, 3, 224, 224)
    return batch_tensor

def main():
    # Source image path
    source_img = r"C:\Users\earlc\image-visualization\cat.jpg"
    dest_img = "cat.jpg"
    
    if not os.path.exists(dest_img) and os.path.exists(source_img):
        print(f"Copying {source_img} to local directory as {dest_img}...")
        shutil.copy(source_img, dest_img)
        
    image_path = dest_img if os.path.exists(dest_img) else source_img
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}!")
        return

    # --- 1. PYTORCH INFERENCE ---
    print("\n--- Running PyTorch Inference ---")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    
    input_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(input_tensor)
        
    # Get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    categories = weights.meta["categories"]
    print("PyTorch Top 5 Predictions:")
    for i in range(5):
        label = categories[top5_catid[i]]
        prob = top5_prob[i].item()
        print(f"  {i+1}: {label} - {prob:.2%}")

    # --- 2. ONNX RUNTIME INFERENCE ---
    onnx_path = os.path.join("public", "models", "resnet50.onnx")
    if not os.path.exists(onnx_path):
        print(f"\n[Warning] ONNX model not found at {onnx_path}. Please run export_onnx.py first.")
        return
        
    print("\n--- Running ONNX Runtime Inference ---")
    try:
        import onnxruntime as ort
        
        # Create ONNX session
        ort_sess = ort.InferenceSession(onnx_path)
        
        # Prepare input data as a numpy array
        onnx_input = input_tensor.numpy()
        
        # Run inference
        outputs = ort_sess.run(["output"], {"input": onnx_input})
        onnx_output = outputs[0]
        
        # Softmax in numpy
        exp_out = np.exp(onnx_output[0] - np.max(onnx_output[0]))
        onnx_probs = exp_out / np.sum(exp_out)
        
        # Get top 5 indices
        top5_idx = np.argsort(onnx_probs)[-5:][::-1]
        
        print("ONNX Runtime Top 5 Predictions:")
        for i, idx in enumerate(top5_idx):
            label = categories[idx]
            prob = onnx_probs[idx]
            print(f"  {i+1}: {label} - {prob:.2%}")
            
        # Compare outputs
        difference = np.max(np.abs(output.numpy() - onnx_output))
        print(f"\nMax difference between PyTorch and ONNX predictions: {difference:.6f}")
        if difference < 1e-4:
            print("SUCCESS: ONNX outputs match PyTorch outputs within acceptable float threshold!")
        else:
            print("WARNING: Output mismatch detected.")
            
    except ImportError:
        print("onnxruntime is not installed. To test ONNX inference, run: pip install onnxruntime")

if __name__ == "__main__":
    main()
