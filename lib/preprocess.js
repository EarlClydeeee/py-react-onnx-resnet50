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

export function validateSkinPhoto(image) {
  const pixels = getImagePixels(image);
  const pixelCount = pixels.length / 4;
  let skinLike = 0;
  let tooDark = 0;
  let tooBright = 0;
  let totalSaturation = 0;
  let totalBrightness = 0;

  for (let i = 0; i < pixelCount; i++) {
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const brightness = (r + g + b) / 3;
    const saturation = max === 0 ? 0 : (max - min) / max;
    const redDominant = r > b * 0.8 && r > g * 0.65;
    const balancedTone = max - min > 0.04 && max - min < 0.65;
    const plausibleExposure = brightness > 0.12 && brightness < 0.95;

    if (redDominant && balancedTone && plausibleExposure) skinLike += 1;
    if (brightness < 0.08) tooDark += 1;
    if (brightness > 0.96) tooBright += 1;
    totalSaturation += saturation;
    totalBrightness += brightness;
  }

  const skinRatio = skinLike / pixelCount;
  const darkRatio = tooDark / pixelCount;
  const brightRatio = tooBright / pixelCount;
  const avgSaturation = totalSaturation / pixelCount;
  const avgBrightness = totalBrightness / pixelCount;
  const qualityScore = clamp01(
    skinRatio * 1.45 +
      (1 - darkRatio) * 0.2 +
      (1 - brightRatio) * 0.15 +
      clamp01(avgSaturation * 2) * 0.1
  );

  if (darkRatio > 0.65) {
    return {
      status: "invalid",
      score: qualityScore,
      reason: "The image is too dark to evaluate reliably.",
    };
  }

  if (brightRatio > 0.65) {
    return {
      status: "invalid",
      score: qualityScore,
      reason: "The image is overexposed, so skin details are not visible.",
    };
  }

  if (skinRatio < 0.18 || avgSaturation < 0.03 || avgBrightness < 0.12) {
    return {
      status: "invalid",
      score: qualityScore,
      reason:
        "This does not look like a close-up skin or lesion photo. Please upload a clear dermoscopy or skin image.",
    };
  }

  if (skinRatio < 0.3) {
    return {
      status: "warning",
      score: qualityScore,
      reason:
        "Only part of the image appears skin-like. Results may be unreliable unless the lesion fills the frame.",
    };
  }

  return {
    status: "valid",
    score: qualityScore,
    reason: "The photo appears suitable for skin lesion classification.",
  };
}
