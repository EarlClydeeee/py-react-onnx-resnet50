"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getTopPredictions,
  preprocessImage,
  validateSkinPhoto,
} from "@/lib/preprocess";
import {
  BoltIcon,
  CpuIcon,
  SparklesIcon,
  UploadIcon,
} from "./icons";

const MODEL_PATH = "/models/resnet50_skin.onnx";
const LABELS_PATH = "/data/skin_labels.json";
const MIN_DISEASE_CONFIDENCE = 0.45;
const MIN_CONFIDENCE_MARGIN = 0.12;

const FEATURES = [
  {
    icon: CpuIcon,
    title: "Browser ONNX Runtime",
    description:
      "ResNet50 runs entirely in your browser via WebAssembly — no server round-trip for inference.",
  },
  {
    icon: BoltIcon,
    title: "HAM10000 fine-tuned",
    description:
      "7-class skin disease classification trained on 10,000+ dermoscopy images using transfer learning.",
  },
  {
    icon: SparklesIcon,
    title: "Live predictions",
    description:
      "Upload a skin lesion photo and see the predicted condition with confidence score in real time.",
  },
];

function StatusBadge({ status }) {
  const styles = {
    loading: "bg-secondary/20 text-primary border-secondary/40",
    ready: "bg-cta/15 text-emerald-800 border-cta/40",
    error: "bg-red-100 text-red-800 border-red-200",
  };

  const labels = {
    loading: "Loading model…",
    ready: "Model ready",
    error: "Model unavailable",
  };

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-medium transition-colors duration-200 ${styles[status]}`}
    >
      {status === "loading" && (
        <span
          className="h-2 w-2 rounded-full bg-primary animate-pulse motion-reduce:animate-none"
          aria-hidden="true"
        />
      )}
      {status === "ready" && (
        <span className="h-2 w-2 rounded-full bg-cta" aria-hidden="true" />
      )}
      {labels[status]}
    </span>
  );
}

function PredictionBar({ label, probability, rank, maxProbability }) {
  const width = maxProbability > 0 ? (probability / maxProbability) * 100 : 0;

  return (
    <li
      className="rounded-xl border border-primary/10 bg-white/80 p-4 transition-colors duration-200"
      style={{ animationDelay: `${rank * 80}ms` }}
    >
      <div className="mb-2 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <span className="text-xs font-semibold uppercase tracking-wide text-secondary">
            #{rank + 1}
          </span>
          <p className="truncate font-medium text-text">{label}</p>
        </div>
        <span className="shrink-0 font-mono text-sm font-semibold text-primary">
          {(probability * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-primary/10">
        <div
          className="h-full rounded-full bg-primary transition-all duration-500 ease-out motion-reduce:transition-none"
          style={{ width: `${width}%` }}
          role="progressbar"
          aria-valuenow={Math.round(probability * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${label}: ${(probability * 100).toFixed(1)}%`}
        />
      </div>
    </li>
  );
}

function ValidationCard({ result }) {
  if (!result) return null;

  const styles = {
    valid: "border-cta/40 bg-cta/10 text-emerald-900",
    warning: "border-amber-300 bg-amber-50 text-amber-900",
    invalid: "border-red-200 bg-red-50 text-red-800",
  };

  const titles = {
    valid: "Skin photo check passed",
    warning: "Skin photo check needs review",
    invalid: "Photo is not suitable",
  };

  return (
    <div className={`mt-4 rounded-2xl border px-4 py-3 ${styles[result.status]}`}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-medium">{titles[result.status]}</p>
          <p className="mt-1 text-sm">{result.reason}</p>
        </div>
        <span className="shrink-0 font-mono text-sm">
          {Math.round(result.score * 100)}%
        </span>
      </div>
    </div>
  );
}

export default function ImageClassifier() {
  const fileInputRef = useRef(null);
  const previewRef = useRef(null);
  const sessionRef = useRef(null);
  const ortRef = useRef(null);

  const [modelStatus, setModelStatus] = useState("loading");
  const [modelError, setModelError] = useState("");
  const [labels, setLabels] = useState([]);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [validationResult, setValidationResult] = useState(null);
  const [reliabilityWarning, setReliabilityWarning] = useState("");
  const [inferenceError, setInferenceError] = useState("");
  const [inferenceMs, setInferenceMs] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const labelsResponse = await fetch(LABELS_PATH);
        if (!labelsResponse.ok) {
          throw new Error("Failed to load ImageNet labels.");
        }
        const labelData = await labelsResponse.json();
        if (!cancelled) setLabels(labelData);

        const ort = await import("onnxruntime-web");
        ort.env.wasm.wasmPaths =
          "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        const modelResponse = await fetch(MODEL_PATH, { method: "HEAD" });
        if (!modelResponse.ok) {
          throw new Error(
            "ONNX model not found. Export from Colab and place resnet50_skin.onnx in public/models/."
          );
        }

        const session = await ort.InferenceSession.create(MODEL_PATH, {
          executionProviders: ["wasm"],
        });

        if (cancelled) return;

        ortRef.current = ort;
        sessionRef.current = session;
        setModelStatus("ready");
      } catch (error) {
        if (!cancelled) {
          setModelStatus("error");
          setModelError(
            error instanceof Error ? error.message : "Failed to load model."
          );
        }
      }
    }

    init();

    return () => {
      cancelled = true;
    };
  }, []);

  const clearPreview = useCallback(() => {
    if (previewUrl?.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setPredictions([]);
    setValidationResult(null);
    setReliabilityWarning("");
    setInferenceError("");
    setInferenceMs(null);
  }, [previewUrl]);

  const handleFile = useCallback(
    (file) => {
      if (!file?.type.startsWith("image/")) return;

      clearPreview();
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setPredictions([]);
      setValidationResult(null);
      setReliabilityWarning("");
      setInferenceError("");
      setInferenceMs(null);
    },
    [clearPreview]
  );

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      setIsDragging(false);
      const file = event.dataTransfer.files?.[0];
      handleFile(file);
    },
    [handleFile]
  );

  const runInference = useCallback(async () => {
    const imageEl = previewRef.current;
    const session = sessionRef.current;
    const ort = ortRef.current;

    if (!imageEl || !session || !ort || modelStatus !== "ready") return;

    setIsInferring(true);
    setInferenceError("");
    setPredictions([]);
    setReliabilityWarning("");
    setInferenceMs(null);

    try {
      const skinCheck = validateSkinPhoto(imageEl);
      setValidationResult(skinCheck);
      if (skinCheck.status === "invalid") {
        setInferenceError(skinCheck.reason);
        return;
      }

      const float32Data = preprocessImage(imageEl);
      const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);

      const start = performance.now();
      const outputs = await session.run({ input: inputTensor });
      const elapsed = Math.round(performance.now() - start);

      const logits = Array.from(outputs.output.data);
      const top5 = getTopPredictions(logits, labels, 5);
      const confidenceMargin =
        (top5[0]?.probability ?? 0) - (top5[1]?.probability ?? 0);

      if (
        (top5[0]?.probability ?? 0) < MIN_DISEASE_CONFIDENCE ||
        confidenceMargin < MIN_CONFIDENCE_MARGIN ||
        skinCheck.status === "warning"
      ) {
        setReliabilityWarning(
          "This image may not contain a clear skin disease pattern. Treat the prediction as low-confidence and upload a clearer lesion close-up if available."
        );
      }

      setPredictions(top5);
      setInferenceMs(elapsed);
    } catch (error) {
      setInferenceError(
        error instanceof Error ? error.message : "Inference failed."
      );
    } finally {
      setIsInferring(false);
    }
  }, [labels, modelStatus]);

  const maxProbability = predictions[0]?.probability ?? 0;

  return (
    <div className="min-h-full bg-background text-text">
      <header className="sticky top-4 z-10 mx-4 rounded-2xl border border-primary/10 bg-white/90 px-5 py-4 shadow-sm backdrop-blur-sm">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-white">
              <CpuIcon className="w-5 h-5" />
            </div>
            <div>
              <p className="font-display text-lg font-bold leading-tight text-text">
                Skin Disease Classifier
              </p>
              <p className="text-sm text-slate-600">ResNet50 · HAM10000 · 7 classes</p>
            </div>
          </div>
          <StatusBadge status={modelStatus} />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 pb-16 pt-8">
        <section className="mb-10 text-center">
          <h1 className="font-display text-4xl font-bold tracking-tight text-text sm:text-5xl">
            Skin Disease Classification with{" "}
            <span className="text-primary">ResNet50</span>
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-slate-600">
            Upload a dermoscopy image and classify it into one of 7 skin
            conditions — running entirely in your browser via ONNX Runtime.
          </p>
        </section>

        {modelStatus === "error" && (
          <div
            className="mb-8 rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-red-800"
            role="alert"
          >
            <p className="font-medium">Model could not be loaded</p>
            <p className="mt-1 text-sm">{modelError}</p>
          </div>
        )}

        <div className="grid gap-8 lg:grid-cols-2">
          <section aria-label="Image upload">
            <div
              className={`relative flex min-h-[320px] flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 transition-colors duration-200 ${
                isDragging
                  ? "border-primary bg-primary/5"
                  : "border-primary/25 bg-white/80 hover:border-primary/50"
              } ${previewUrl ? "border-solid" : ""}`}
              onDragEnter={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setIsDragging(false);
              }}
              onDrop={onDrop}
            >
              {previewUrl ? (
                <div className="flex w-full flex-col items-center gap-4">
                  <div className="relative aspect-square w-full max-w-sm overflow-hidden rounded-xl border border-primary/10 bg-slate-100">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      ref={previewRef}
                      src={previewUrl}
                      alt="Uploaded image preview"
                      className="h-full w-full object-contain"
                      onLoad={() => {
                        if (previewRef.current) {
                          previewRef.current.decode?.();
                          setValidationResult(
                            validateSkinPhoto(previewRef.current)
                          );
                        }
                      }}
                    />
                  </div>
                  <ValidationCard result={validationResult} />
                  <div className="flex flex-wrap items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="cursor-pointer rounded-full border border-primary/20 px-4 py-2 text-sm font-medium text-primary transition-colors duration-200 hover:bg-primary/5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                    >
                      Change image
                    </button>
                    <button
                      type="button"
                      onClick={clearPreview}
                      className="cursor-pointer rounded-full px-4 py-2 text-sm font-medium text-slate-600 transition-colors duration-200 hover:bg-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex cursor-pointer flex-col items-center gap-4 text-center focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary"
                >
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                    <UploadIcon className="w-8 h-8" />
                  </div>
                  <div>
                    <p className="font-display text-xl font-semibold text-text">
                      Drop an image here
                    </p>
                    <p className="mt-1 text-sm text-slate-600">
                      or click to browse — JPG, PNG, WebP
                    </p>
                  </div>
                </button>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
            </div>

            <button
              type="button"
              disabled={
                !previewUrl || modelStatus !== "ready" || isInferring
              }
              onClick={runInference}
              className="mt-6 flex w-full cursor-pointer items-center justify-center gap-2 rounded-2xl bg-cta px-6 py-4 font-display text-lg font-semibold text-white shadow-md shadow-cta/25 transition-colors duration-200 hover:bg-emerald-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cta disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none motion-reduce:transition-none"
            >
              {isInferring ? (
                <>
                  <span
                    className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white motion-reduce:animate-none"
                    aria-hidden="true"
                  />
                  Classifying…
                </>
              ) : (
                <>
                  <SparklesIcon className="w-5 h-5" />
                  Classify image
                </>
              )}
            </button>
          </section>

          <section aria-label="Classification results">
            <div className="rounded-2xl border border-primary/10 bg-white/80 p-6 shadow-sm">
              <div className="mb-6 flex items-center justify-between gap-4">
                <h2 className="font-display text-2xl font-bold text-text">
                  Top predictions
                </h2>
                {inferenceMs !== null && (
                  <span className="font-mono text-sm text-slate-600">
                    {inferenceMs} ms
                  </span>
                )}
              </div>

              {isInferring && (
                <div className="space-y-3" aria-live="polite" aria-busy="true">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div
                      key={i}
                      className="h-16 animate-pulse rounded-xl bg-primary/10 motion-reduce:animate-none"
                    />
                  ))}
                </div>
              )}

              {!isInferring && inferenceError && (
                <p className="text-red-700" role="alert">{inferenceError}</p>
              )}

              {!isInferring && reliabilityWarning && (
                <div
                  className="mb-4 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900"
                  role="status"
                >
                  <p className="font-medium">Prediction reliability warning</p>
                  <p className="mt-1">{reliabilityWarning}</p>
                </div>
              )}

              {!isInferring && !inferenceError && predictions.length === 0 && (
                <div className="flex flex-col items-center justify-center py-12 text-center text-slate-600">
                  <SparklesIcon className="mb-3 w-10 h-10 text-secondary/60" />
                  <p>Upload an image and run classification to see results.</p>
                </div>
              )}

              {!isInferring && predictions.length > 0 && (
                <ul className="space-y-3" aria-live="polite">
                  {predictions.map((prediction, index) => (
                    <PredictionBar
                      key={prediction.index}
                      rank={index}
                      label={prediction.label}
                      probability={prediction.probability}
                      maxProbability={maxProbability}
                    />
                  ))}
                </ul>
              )}
            </div>
          </section>
        </div>

        <section className="mt-16 grid gap-6 sm:grid-cols-3" aria-label="Features">
          {FEATURES.map(({ icon: Icon, title, description }) => (
            <article
              key={title}
              className="rounded-2xl border border-primary/10 bg-white/80 p-6 transition-colors duration-200 hover:border-primary/25"
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <Icon className="w-5 h-5" />
              </div>
              <h3 className="font-display text-lg font-semibold text-text">
                {title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                {description}
              </p>
            </article>
          ))}
        </section>

        <section className="mt-10 rounded-2xl border border-primary/10 bg-primary/5 p-6">
          <h2 className="font-display text-lg font-semibold text-text">
            Skin disease classes
          </h2>
          <p className="mt-2 text-sm text-slate-600">
            This model classifies dermoscopy images into 7 categories from the HAM10000 dataset.
          </p>
          <ul className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-sm text-slate-700 sm:grid-cols-4">
            {[
              "Actinic Keratosis",
              "Basal Cell Carcinoma",
              "Benign Keratosis",
              "Dermatofibroma",
              "Melanoma",
              "Melanocytic Nevi",
              "Vascular Lesions",
            ].map((cls) => (
              <li key={cls} className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-primary/60 shrink-0" />
                {cls}
              </li>
            ))}
          </ul>
        </section>
      </main>
    </div>
  );
}
