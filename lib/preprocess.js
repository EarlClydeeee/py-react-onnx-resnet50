const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const INPUT_SIZE = 224;
const RESIZE_SIZE = 256;

function resizeShortEdge(width, height, targetSize) {
  const shortEdge = Math.min(width, height);
  const scale = targetSize / shortEdge;
  return {
    width: Math.round(width * scale),
    height: Math.round(height * scale),
  };
}

/**
 * Preprocess an image for ResNet50 ImageNet inference.
 * Matches torchvision: Resize(256) → CenterCrop(224) → Normalize.
 */
export function preprocessImage(image) {
  const { width: resizedWidth, height: resizedHeight } = resizeShortEdge(
    image.naturalWidth,
    image.naturalHeight,
    RESIZE_SIZE
  );

  const resizeCanvas = document.createElement("canvas");
  resizeCanvas.width = resizedWidth;
  resizeCanvas.height = resizedHeight;
  const resizeCtx = resizeCanvas.getContext("2d");
  resizeCtx.drawImage(image, 0, 0, resizedWidth, resizedHeight);

  const cropX = Math.floor((resizedWidth - INPUT_SIZE) / 2);
  const cropY = Math.floor((resizedHeight - INPUT_SIZE) / 2);

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = INPUT_SIZE;
  cropCanvas.height = INPUT_SIZE;
  const cropCtx = cropCanvas.getContext("2d");
  cropCtx.drawImage(
    resizeCanvas,
    cropX,
    cropY,
    INPUT_SIZE,
    INPUT_SIZE,
    0,
    0,
    INPUT_SIZE,
    INPUT_SIZE
  );

  const { data: pixels } = cropCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const pixelCount = INPUT_SIZE * INPUT_SIZE;

  for (let i = 0; i < pixelCount; i++) {
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    float32Data[i] = (r - MEAN[0]) / STD[0];
    float32Data[pixelCount + i] = (g - MEAN[1]) / STD[1];
    float32Data[2 * pixelCount + i] = (b - MEAN[2]) / STD[2];
  }

  return float32Data;
}

export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

export function getTopPredictions(logits, labels, topK = 5) {
  const probabilities = softmax(logits);
  const ranked = probabilities
    .map((probability, index) => ({
      index,
      label: labels[index] ?? `class_${index}`,
      probability,
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, topK);

  return ranked;
}
