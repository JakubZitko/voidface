// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Voidface contributors
//
// Voidface browser demo — runs the shipped generator via
// onnxruntime-web with the WebGPU execution provider (WASM fallback).
//
// Everything happens in-tab. Files are never uploaded.

import * as ort from "onnxruntime-web";

const modelInput = document.querySelector<HTMLInputElement>("#model")!;
const modelStatus = document.querySelector<HTMLElement>("#modelStatus")!;
const drop = document.querySelector<HTMLElement>("#drop")!;
const picker = document.querySelector<HTMLInputElement>("#picker")!;
const panels = document.querySelector<HTMLElement>("#panels")!;
const originalImg = document.querySelector<HTMLImageElement>("#original")!;
const protectedImg = document.querySelector<HTMLImageElement>("#protected")!;
const statusEl = document.querySelector<HTMLElement>("#status")!;

let session: ort.InferenceSession | null = null;

function say(msg: string) {
  statusEl.textContent = `Status: ${msg}`;
}

async function loadModelFromFile(file: File) {
  say(`loading ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)...`);
  const buffer = await file.arrayBuffer();
  const providers = getPreferredProviders();
  session = await ort.InferenceSession.create(buffer, {
    executionProviders: providers,
  });
  modelStatus.textContent = ` ✓ ${file.name} loaded (${providers.join(" / ")})`;
  say(`model ready. Drop a photo.`);
}

function getPreferredProviders(): string[] {
  // Prefer WebGPU; fall back to WASM.
  if ("gpu" in navigator) return ["webgpu", "wasm"];
  return ["wasm"];
}

modelInput.addEventListener("change", async (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    await loadModelFromFile(file);
  } catch (error) {
    modelStatus.textContent = " ✗ failed to load";
    say(`error: ${(error as Error).message}`);
  }
});

drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("drag");
});
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", async (event) => {
  event.preventDefault();
  drop.classList.remove("drag");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  await protectImage(file);
});
picker.addEventListener("change", async (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  await protectImage(file);
});

async function protectImage(file: File) {
  if (!session) {
    say("no model loaded — pick a .ort file first.");
    return;
  }
  if (!file.type.startsWith("image/")) {
    say("that does not look like an image file.");
    return;
  }
  say(`decoding ${file.name}...`);
  const imageBitmap = await createImageBitmap(file);
  const { width, height } = padToDivisor(imageBitmap, 16);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(imageBitmap, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);

  const tensor = imageDataToFloatTensor(imageData);
  say(`running generator on ${width}x${height}...`);
  const start = performance.now();
  const outputs = await session.run({ input: tensor });
  const elapsed = performance.now() - start;

  const outputTensor = outputs[Object.keys(outputs)[0]];
  const outputData = floatTensorToImageData(outputTensor, width, height);

  originalImg.src = URL.createObjectURL(file);
  const outCanvas = new OffscreenCanvas(width, height);
  outCanvas.getContext("2d")!.putImageData(outputData, 0, 0);
  const outBlob = await outCanvas.convertToBlob({ type: "image/png" });
  protectedImg.src = URL.createObjectURL(outBlob);
  panels.style.display = "grid";
  say(`done in ${elapsed.toFixed(0)} ms.`);
}

function padToDivisor(source: ImageBitmap, divisor: number) {
  const width = Math.ceil(source.width / divisor) * divisor;
  const height = Math.ceil(source.height / divisor) * divisor;
  return { width, height };
}

function imageDataToFloatTensor(image: ImageData): ort.Tensor {
  const { width, height, data } = image;
  const size = width * height;
  const array = new Float32Array(1 * 3 * size);
  for (let i = 0; i < size; i++) {
    array[i] = data[i * 4] / 255;
    array[size + i] = data[i * 4 + 1] / 255;
    array[size * 2 + i] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", array, [1, 3, height, width]);
}

function floatTensorToImageData(tensor: ort.Tensor, width: number, height: number): ImageData {
  const data = tensor.data as Float32Array;
  const size = width * height;
  const rgba = new Uint8ClampedArray(size * 4);
  for (let i = 0; i < size; i++) {
    rgba[i * 4] = Math.round(clamp(data[i]) * 255);
    rgba[i * 4 + 1] = Math.round(clamp(data[size + i]) * 255);
    rgba[i * 4 + 2] = Math.round(clamp(data[size * 2 + i]) * 255);
    rgba[i * 4 + 3] = 255;
  }
  return new ImageData(rgba, width, height);
}

function clamp(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

say("waiting for model — pick a .ort or .onnx file above.");
