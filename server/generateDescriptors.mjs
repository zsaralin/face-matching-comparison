import fs from 'fs';
import path from 'path';
import * as faceapi from '@vladmandic/face-api';
import canvas from 'canvas';
import { fileURLToPath } from 'url';

// Required for face-api.js in Node
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const outputFile = 'descriptors.ndjson';

async function loadModels() {
  const modelPath = path.join(process.cwd(), 'models');
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
}

async function generateDescriptor(imagePath) {
  const img = await canvas.loadImage(imagePath);
  const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
  if (!detections) return null;
  return Array.from(detections.descriptor); // convert Float32Array to plain array
}

async function processFolder(root) {
  const subfolders = fs.readdirSync(root, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name);

  const output = fs.createWriteStream(outputFile, { flags: 'w' });

  for (const folder of subfolders) {
    const imageDir = path.join(root, folder, 'images');
    if (!fs.existsSync(imageDir)) continue;

    const files = fs.readdirSync(imageDir);
    const cmpFile = files.find(f => f.endsWith('cmp.png'));
    if (!cmpFile) {
      console.warn(`⚠️  No cmp.png in ${folder}`);
      continue;
    }

    const imagePath = path.join(imageDir, cmpFile);
    const descriptor = await generateDescriptor(imagePath);

    if (descriptor) {
      const json = JSON.stringify({ key: folder, embedding: descriptor });
      output.write(json + '\n');
      console.log(`✅ Processed ${folder}`);
    } else {
      console.warn(`❌ Face not detected in ${folder}`);
    }
  }

  output.end();
}

(async () => {
  console.log('🔄 Loading models...');
  await loadModels();
  console.log('🚀 Starting processing...');
  await processFolder('../db/small');
  console.log('✅ Done. Output saved to', outputFile);
})();
