const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const INPUT_SIZE = 224;
const RESIZE_SIZE = 256;
const VALIDATION_SIZE = 160;

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function getImagePixels(image, size = VALIDATION_SIZE) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size).data;
}

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

/**
 * Validate that an uploaded image looks like a brain MRI scan.
 * MRI scans are typically grayscale (low saturation), moderate brightness,
 * and show good contrast between brain tissue and background.
 */
export function validateBrainMRI(image) {
  const aspectRatio = image.naturalWidth / image.naturalHeight;
  const pixels = getImagePixels(image);
  const pixelCount = pixels.length / 4;

  let tooDark = 0;
  let tooBright = 0;
  let totalSaturation = 0;
  let totalBrightness = 0;
  let contrastSum = 0;

  for (let i = 0; i < pixelCount; i++) {
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const brightness = (r + g + b) / 3;
    const saturation = max === 0 ? 0 : (max - min) / max;

    totalSaturation += saturation;
    totalBrightness += brightness;
    contrastSum += max - min;

    if (brightness < 0.05) tooDark += 1;
    if (brightness > 0.97) tooBright += 1;
  }

  const darkRatio = tooDark / pixelCount;
  const brightRatio = tooBright / pixelCount;
  const avgSaturation = totalSaturation / pixelCount;
  const avgBrightness = totalBrightness / pixelCount;
  const avgContrast = contrastSum / pixelCount;

  // Quality score: reward low saturation (grayscale), good brightness range, good contrast
  const qualityScore = clamp01(
    (1 - clamp01(avgSaturation * 4)) * 0.4 +
      clamp01(avgBrightness * 3) * 0.2 +
      (1 - darkRatio) * 0.2 +
      (1 - brightRatio) * 0.1 +
      clamp01(avgContrast * 5) * 0.1
  );

  if (aspectRatio > 1.6 || aspectRatio < 0.6) {
    return {
      status: "invalid",
      score: qualityScore,
      reason:
        "This looks like a report figure or composite image. Upload one single brain MRI scan, not a multi-image chart or screenshot.",
    };
  }

  if (darkRatio > 0.80) {
    return {
      status: "invalid",
      score: qualityScore,
      reason: "The image is too dark. Please upload a properly exposed MRI scan.",
    };
  }

  if (brightRatio > 0.70) {
    return {
      status: "invalid",
      score: qualityScore,
      reason: "The image is overexposed. Please upload a properly exposed MRI scan.",
    };
  }

  if (avgBrightness < 0.05) {
    return {
      status: "invalid",
      score: qualityScore,
      reason: "The image appears nearly black. Please upload a brain MRI scan.",
    };
  }

  // Highly saturated image is likely a color photo, not an MRI
  if (avgSaturation > 0.35) {
    return {
      status: "invalid",
      score: qualityScore,
      reason:
        "This looks like a color photograph, not a brain MRI scan. Please upload a grayscale MRI image.",
    };
  }

  // Mildly colorful images are blocked because color photos and screenshots can still
  // produce confident but meaningless tumor predictions.
  if (avgSaturation > 0.18) {
    return {
      status: "invalid",
      score: qualityScore,
      reason:
        "This image has more color than a typical MRI scan. Upload a single grayscale brain MRI image for reliable classification.",
    };
  }

  return {
    status: "valid",
    score: qualityScore,
    reason: "Image appears suitable for brain tumor classification.",
  };
}
