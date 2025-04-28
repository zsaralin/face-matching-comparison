const fs = require('fs');
const path = require('path');
const faceapi = require('@vladmandic/face-api');
const canvas = require('canvas');
const { Buffer } = require('buffer');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const MODEL_PATH = path.join(process.cwd(), 'models');
const NDJSON_PATH = path.join(process.cwd(), 'descriptors.ndjson');

// Load models once
let modelsLoaded = false;
async function loadModelsOnce() {
  if (modelsLoaded) return;
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
  modelsLoaded = true;
}

async function getDescriptor(base64Image) {
  await loadModelsOnce();

  const base64Data = base64Image.replace(/^data:image\/\w+;base64,/, '');
  const imgBuffer = Buffer.from(base64Data, 'base64');
  const img = await canvas.loadImage(imgBuffer);

  const detection = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();

  return detection ? Array.from(detection.descriptor) : null;
}

function euclideanDistance(a, b) {
  return Math.sqrt(a.reduce((acc, val, i) => acc + (val - b[i]) ** 2, 0));
}

async function compareDescriptor(inputDescriptor, topN = 10) {
  const lines = fs.readFileSync(NDJSON_PATH, 'utf8').split('\n').filter(Boolean);

  const results = [];

  for (const line of lines) {
    const { key, embedding } = JSON.parse(line);
    const distance = euclideanDistance(inputDescriptor, embedding);

    const filePath = path.posix.join('db', 'small', key, 'images', `${key}_cmp.png`);

    results.push({ key, file: filePath, distance });
  }

  results.sort((a, b) => a.distance - b.distance);
  return results.slice(0, topN);
}

module.exports = {
  loadModelsOnce,
  getDescriptor,
  compareDescriptor,
};
