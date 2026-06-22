"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getTopPredictions,
  preprocessImage,
  validateBrainMRI,
} from "@/lib/preprocess";

// ── SVG Icons ─────────────────────────────────────────────────────────────────
function IconBrain({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />
    </svg>
  );
}

function IconUpload({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function IconScan({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 7V5a2 2 0 0 1 2-2h2" />
      <path d="M17 3h2a2 2 0 0 1 2 2v2" />
      <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
      <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
      <line x1="7" y1="12" x2="17" y2="12" />
    </svg>
  );
}

function IconShield({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}

function IconZap({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

function IconCheck({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconAlertTriangle({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function IconX({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function IconImage({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </svg>
  );
}

// ── Constants ─────────────────────────────────────────────────────────────────
const MODEL_PATH = "/models/resnet50_brain.onnx";
const LABELS_PATH = "/data/brain_labels.json";
const MIN_DISEASE_CONFIDENCE = 0.45;
const MIN_CONFIDENCE_MARGIN = 0.12;

const TUMOR_CLASSES = [
  { name: "Glioma Tumor", desc: "Most common malignant primary brain tumor" },
  { name: "Meningioma Tumor", desc: "Arises from the meninges surrounding the brain" },
  { name: "No Tumor", desc: "Normal brain tissue — no pathology detected" },
  { name: "Pituitary Tumor", desc: "Occurs in the pituitary gland at the base of the skull" },
];

const FEATURES = [
  {
    icon: IconZap,
    title: "100% In-Browser",
    description: "ResNet50 runs entirely via WebAssembly — your MRI never leaves your device. No server, no upload.",
  },
  {
    icon: IconScan,
    title: "4-Class MRI Analysis",
    description: "Trained on 7,000+ labeled brain MRI scans using transfer learning from ImageNet.",
  },
  {
    icon: IconShield,
    title: "MRI Validation",
    description: "Automatically checks whether the uploaded image is a valid grayscale brain MRI before classifying.",
  },
];

// ── Sub-components ─────────────────────────────────────────────────────────────
function StatusBadge({ status }) {
  if (status === "loading") {
    return (
      <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm font-medium text-slate-600">
        <span className="h-2 w-2 rounded-full bg-slate-400 animate-pulse motion-reduce:animate-none" aria-hidden="true" />
        Loading model…
      </span>
    );
  }
  if (status === "ready") {
    return (
      <span className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-sm font-medium text-emerald-800">
        <span className="h-2 w-2 rounded-full bg-emerald-500" aria-hidden="true" />
        Model ready
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-red-200 bg-red-50 px-3 py-1.5 text-sm font-medium text-red-700">
      <span className="h-2 w-2 rounded-full bg-red-500" aria-hidden="true" />
      Unavailable
    </span>
  );
}

function ValidationCard({ result }) {
  if (!result) return null;

  if (result.status === "valid") {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3">
        <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-500">
          <IconCheck className="w-3 h-3 text-white" />
        </div>
        <div>
          <p className="text-sm font-semibold text-emerald-800">MRI check passed</p>
          <p className="mt-0.5 text-xs text-emerald-700">{result.reason}</p>
        </div>
        <span className="ml-auto shrink-0 font-mono text-xs text-emerald-600">{Math.round(result.score * 100)}%</span>
      </div>
    );
  }

  if (result.status === "warning") {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
        <IconAlertTriangle className="mt-0.5 w-5 h-5 shrink-0 text-amber-600" />
        <div>
          <p className="text-sm font-semibold text-amber-800">Needs review</p>
          <p className="mt-0.5 text-xs text-amber-700">{result.reason}</p>
        </div>
        <span className="ml-auto shrink-0 font-mono text-xs text-amber-600">{Math.round(result.score * 100)}%</span>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3">
      <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-red-500">
        <IconX className="w-3 h-3 text-white" />
      </div>
      <div>
        <p className="text-sm font-semibold text-red-800">Image not suitable</p>
        <p className="mt-0.5 text-xs text-red-700">{result.reason}</p>
      </div>
    </div>
  );
}

function PredictionBar({ label, probability, rank, maxProbability, isTop }) {
  const width = maxProbability > 0 ? (probability / maxProbability) * 100 : 0;
  const pct = (probability * 100).toFixed(1);

  return (
    <li className={`rounded-xl border p-4 transition-colors duration-200 ${
      isTop
        ? "border-primary/30 bg-primary/5"
        : "border-slate-100 bg-white"
    }`}>
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
            isTop ? "bg-primary text-white" : "bg-slate-100 text-slate-500"
          }`}>
            {rank + 1}
          </span>
          <p className={`truncate font-semibold ${isTop ? "text-primary" : "text-slate-700"}`}>
            {label}
          </p>
        </div>
        <span className={`shrink-0 font-mono text-sm font-bold ${isTop ? "text-primary" : "text-slate-500"}`}>
          {pct}%
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out motion-reduce:transition-none ${
            isTop ? "bg-primary" : "bg-slate-300"
          }`}
          style={{ width: `${width}%` }}
          role="progressbar"
          aria-valuenow={Math.round(probability * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${label}: ${pct}%`}
        />
      </div>
    </li>
  );
}

function SkeletonBar() {
  return (
    <div className="h-[72px] animate-pulse rounded-xl bg-slate-100 motion-reduce:animate-none" />
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────
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
        if (!labelsResponse.ok) throw new Error("Failed to load class labels.");
        const labelData = await labelsResponse.json();
        if (!cancelled) setLabels(labelData);

        const ort = await import("onnxruntime-web");
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        const modelResponse = await fetch(MODEL_PATH, { method: "HEAD" });
        if (!modelResponse.ok) {
          throw new Error("ONNX model not found. Place resnet50_brain.onnx in public/models/.");
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
          setModelError(error instanceof Error ? error.message : "Failed to load model.");
        }
      }
    }

    init();
    return () => { cancelled = true; };
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

  const handleFile = useCallback((file) => {
    if (!file?.type.startsWith("image/")) return;
    clearPreview();
    setPreviewUrl(URL.createObjectURL(file));
    setPredictions([]);
    setValidationResult(null);
    setReliabilityWarning("");
    setInferenceError("");
    setInferenceMs(null);
  }, [clearPreview]);

  const onDrop = useCallback((event) => {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  }, [handleFile]);

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
      const mriCheck = validateBrainMRI(imageEl);
      setValidationResult(mriCheck);
      if (mriCheck.status !== "valid") {
        setInferenceError(mriCheck.reason);
        return;
      }

      const float32Data = preprocessImage(imageEl);
      const inputTensor = new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);

      const start = performance.now();
      const outputs = await session.run({ input: inputTensor });
      const elapsed = Math.round(performance.now() - start);

      const logits = Array.from(outputs.output.data);
      const top4 = getTopPredictions(logits, labels, 4);
      const margin = (top4[0]?.probability ?? 0) - (top4[1]?.probability ?? 0);

      if ((top4[0]?.probability ?? 0) < MIN_DISEASE_CONFIDENCE || margin < MIN_CONFIDENCE_MARGIN) {
        setReliabilityWarning(
          "Confidence is low — the model is uncertain. Results should not be used for clinical decisions. Try uploading a higher-quality single MRI scan."
        );
      }

      setPredictions(top4);
      setInferenceMs(elapsed);
    } catch (error) {
      setInferenceError(error instanceof Error ? error.message : "Inference failed.");
    } finally {
      setIsInferring(false);
    }
  }, [labels, modelStatus]);

  const maxProbability = predictions[0]?.probability ?? 0;
  const canClassify = !!previewUrl && modelStatus === "ready" && !isInferring;

  return (
    <div className="min-h-full bg-background text-text">

      {/* ── Navbar ── */}
      <header className="sticky top-3 z-10 mx-3 sm:mx-6">
        <div className="mx-auto max-w-6xl rounded-2xl border border-slate-200 bg-white/95 px-4 py-3 shadow-sm backdrop-blur-sm">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-white shadow-sm">
                <IconBrain className="w-5 h-5" />
              </div>
              <div>
                <p className="text-base font-bold leading-tight text-text">
                  Brain Tumor Classifier
                </p>
                <p className="text-xs text-slate-500">ResNet50 · ONNX Runtime · 4 classes</p>
              </div>
            </div>
            <StatusBadge status={modelStatus} />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 pb-20 pt-10">

        {/* ── Hero ── */}
        <section className="mb-12 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/8 px-4 py-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden="true" />
            <span className="text-sm font-medium text-primary">Research demonstration — not for clinical use</span>
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-text sm:text-5xl">
            Brain MRI{" "}
            <span className="text-primary">Tumor Classification</span>
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-lg text-slate-600">
            Upload a single grayscale brain MRI scan. ResNet50 classifies it into one of 4 tumor categories — entirely in your browser.
          </p>
        </section>

        {/* ── Model error banner ── */}
        {modelStatus === "error" && (
          <div className="mb-8 flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-red-800" role="alert">
            <IconX className="mt-0.5 h-5 w-5 shrink-0" />
            <div>
              <p className="font-semibold">Model could not be loaded</p>
              <p className="mt-1 text-sm">{modelError}</p>
            </div>
          </div>
        )}

        {/* ── Main grid ── */}
        <div className="grid gap-6 lg:grid-cols-2">

          {/* Upload panel */}
          <section aria-label="Image upload">
            <div
              className={`relative flex min-h-[340px] flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 transition-colors duration-200 ${
                isDragging
                  ? "border-primary bg-primary/5"
                  : previewUrl
                    ? "border-slate-200 bg-white"
                    : "border-slate-300 bg-white hover:border-primary/50 hover:bg-primary/3"
              }`}
              onDragEnter={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
              onDrop={onDrop}
            >
              {previewUrl ? (
                <div className="flex w-full flex-col gap-4">
                  {/* Image preview */}
                  <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      ref={previewRef}
                      src={previewUrl}
                      alt="Uploaded MRI image preview"
                      className="mx-auto block max-h-64 w-full object-contain"
                      onLoad={() => {
                        if (previewRef.current) {
                          setValidationResult(validateBrainMRI(previewRef.current));
                        }
                      }}
                    />
                  </div>

                  {/* Validation result */}
                  {validationResult && <ValidationCard result={validationResult} />}

                  {/* Action buttons */}
                  <div className="flex items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="cursor-pointer rounded-full border border-primary/25 px-4 py-2 text-sm font-medium text-primary transition-colors duration-200 hover:bg-primary/8 focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-primary"
                    >
                      Change image
                    </button>
                    <button
                      type="button"
                      onClick={clearPreview}
                      className="cursor-pointer rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 transition-colors duration-200 hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-primary"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex cursor-pointer flex-col items-center gap-5 text-center focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-primary rounded-xl p-4"
                >
                  <div className="flex h-20 w-20 items-center justify-center rounded-2xl border-2 border-dashed border-primary/30 bg-primary/8 text-primary transition-colors duration-200 group-hover:bg-primary/15">
                    <IconUpload className="w-9 h-9" />
                  </div>
                  <div>
                    <p className="text-xl font-bold text-text">Drop your MRI scan here</p>
                    <p className="mt-1 text-sm text-slate-500">or click to browse — JPG, PNG, WebP</p>
                    <p className="mt-2 text-xs text-slate-400">Single grayscale scan only · not composite images</p>
                  </div>
                </button>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                aria-label="Upload brain MRI image"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
            </div>

            {/* Classify button */}
            <button
              type="button"
              disabled={!canClassify}
              onClick={runInference}
              className="mt-4 flex min-h-[52px] w-full cursor-pointer items-center justify-center gap-2.5 rounded-2xl bg-cta px-6 py-3.5 text-base font-bold text-white shadow-md shadow-green-500/20 transition-colors duration-200 hover:bg-[#15803d] focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-cta focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none motion-reduce:transition-none"
            >
              {isInferring ? (
                <>
                  <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white motion-reduce:animate-none" aria-hidden="true" />
                  Classifying…
                </>
              ) : (
                <>
                  <IconScan className="w-5 h-5" />
                  Classify MRI
                </>
              )}
            </button>
          </section>

          {/* Results panel */}
          <section aria-label="Classification results">
            <div className="h-full rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

              {/* Header */}
              <div className="mb-5 flex items-center justify-between gap-4 border-b border-slate-100 pb-4">
                <h2 className="text-xl font-bold text-text">Classification Results</h2>
                {inferenceMs !== null && (
                  <span className="rounded-full bg-slate-100 px-2.5 py-1 font-mono text-xs text-slate-500">
                    {inferenceMs} ms
                  </span>
                )}
              </div>

              {/* Loading skeletons */}
              {isInferring && (
                <div className="space-y-3" aria-live="polite" aria-busy="true" aria-label="Classifying image">
                  {[0, 1, 2, 3].map((i) => <SkeletonBar key={i} />)}
                </div>
              )}

              {/* Inference blocked by validation */}
              {!isInferring && inferenceError && (
                <div className="flex flex-col items-center gap-3 py-10 text-center" role="alert">
                  <div className="flex h-14 w-14 items-center justify-center rounded-full bg-red-50 text-red-500">
                    <IconAlertTriangle className="w-7 h-7" />
                  </div>
                  <div>
                    <p className="font-semibold text-slate-800">Cannot classify this image</p>
                    <p className="mt-1 text-sm text-slate-500">{inferenceError}</p>
                  </div>
                </div>
              )}

              {/* Low-confidence warning */}
              {!isInferring && reliabilityWarning && (
                <div className="mb-4 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3" role="status">
                  <IconAlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                  <div>
                    <p className="text-sm font-semibold text-amber-800">Low confidence</p>
                    <p className="mt-0.5 text-xs text-amber-700">{reliabilityWarning}</p>
                  </div>
                </div>
              )}

              {/* Empty state */}
              {!isInferring && !inferenceError && predictions.length === 0 && (
                <div className="flex flex-col items-center justify-center gap-4 py-12 text-center">
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/8 text-primary">
                    <IconImage className="w-8 h-8" />
                  </div>
                  <div>
                    <p className="font-semibold text-slate-700">No results yet</p>
                    <p className="mt-1 text-sm text-slate-500">Upload a brain MRI scan and click Classify MRI.</p>
                  </div>
                </div>
              )}

              {/* Predictions */}
              {!isInferring && predictions.length > 0 && (
                <ul className="space-y-2.5" aria-live="polite" aria-label="Tumor classification predictions">
                  {predictions.map((pred, i) => (
                    <PredictionBar
                      key={pred.index}
                      rank={i}
                      label={pred.label}
                      probability={pred.probability}
                      maxProbability={maxProbability}
                      isTop={i === 0}
                    />
                  ))}
                </ul>
              )}

              {/* Disclaimer */}
              {!isInferring && predictions.length > 0 && (
                <p className="mt-5 text-xs text-slate-400 leading-relaxed border-t border-slate-100 pt-4">
                  For research purposes only. Not intended for clinical diagnosis or medical decision-making.
                </p>
              )}
            </div>
          </section>
        </div>

        {/* ── Feature cards ── */}
        <section className="mt-10 grid gap-4 sm:grid-cols-3" aria-label="Features">
          {FEATURES.map(({ icon: Icon, title, description }) => (
            <article
              key={title}
              className="rounded-2xl border border-slate-200 bg-white p-5 transition-colors duration-200 hover:border-primary/30 hover:shadow-sm cursor-default"
            >
              <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <Icon className="w-5 h-5" />
              </div>
              <h3 className="font-bold text-text">{title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-500">{description}</p>
            </article>
          ))}
        </section>

        {/* ── Tumor class reference ── */}
        <section className="mt-6 rounded-2xl border border-primary/15 bg-primary/5 p-6">
          <div className="mb-4 flex items-center gap-2">
            <IconBrain className="w-5 h-5 text-primary" />
            <h2 className="font-bold text-text">Supported Tumor Classes</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {TUMOR_CLASSES.map(({ name, desc }) => (
              <div key={name} className="flex items-start gap-3 rounded-xl bg-white/70 px-4 py-3 border border-white">
                <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-primary" aria-hidden="true" />
                <div>
                  <p className="text-sm font-semibold text-text">{name}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-slate-200 bg-white/60 py-6 text-center">
        <p className="text-sm text-slate-500">
          ResNet50 · Brain Tumor MRI Dataset · 7,023 images · Transfer Learning · ONNX Runtime Web
        </p>
      </footer>
    </div>
  );
}
