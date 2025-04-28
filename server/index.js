const { loadModelsOnce, getDescriptor } = require('./descriptorMatcher');
const express = require('express');
const cors = require('cors');
const { compareDescriptor } = require('./descriptorMatcher');
const path = require('path');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const fs = require('fs');

const app = express();
const PORT = 5000;

app.use('/db', express.static(path.join(__dirname, '..', 'db')));
app.use(cors());
app.use(express.json({ limit: '10mb' })); // Support large base64 images

// 🟠 FaceAPI (Node.js directly)
app.post('/get-matches', async (req, res) => {
  const { imageData } = req.body;

  if (!imageData) {
    return res.status(400).send('No image data received.');
  }

  try {
    await loadModelsOnce();
    const descriptor = await getDescriptor(imageData);

    if (!descriptor) {
      return res.status(400).send('No face detected.');
    }

    const matches = await compareDescriptor(descriptor, 26);
    res.status(200).json(matches);
  } catch (err) {
    console.error('❌ Error processing snapshot:', err);
    res.status(500).send('Internal error while processing image.');
  }
});

// 🟠 InsightFace (calls insight venv python)
app.post('/get-matches-insight', (req, res) => {
  const { imageData } = req.body;
  if (!imageData) return res.status(400).send('No image data received.');

  const tempFilename = `temp_${uuidv4()}.json`;
  const tempFilePath = path.join(__dirname, tempFilename);

  try {
    fs.writeFileSync(tempFilePath, JSON.stringify({ imageData }));
  } catch (err) {
    console.error('❌ Failed to save JSON file:', err);
    return res.status(500).send('Failed to write JSON file.');
  }

  console.log('📤 Calling InsightFace Python with temp JSON:', tempFilePath);

  const python = spawn(
    path.join(__dirname, '..', 'venv-insight', 'Scripts', 'python.exe'), // 👉 use insight venv
    ['descriptor-matcher.py', tempFilePath]
  );

  handlePythonResponse(python, tempFilePath, res);
});

// 🟠 DeepFace (calls deepface venv python)
app.post('/get-matches-deepface', (req, res) => {
  const { imageData } = req.body;
  if (!imageData) return res.status(400).send('No image data received.');

  const tempFilename = `temp_${uuidv4()}.json`;
  const tempFilePath = path.join(__dirname, tempFilename);

  try {
    fs.writeFileSync(tempFilePath, JSON.stringify({ imageData }));
  } catch (err) {
    console.error('❌ Failed to save JSON file:', err);
    return res.status(500).send('Failed to write JSON file.');
  }

  console.log('📤 Calling DeepFace Python with temp JSON:', tempFilePath);

  const python = spawn(
    path.join(__dirname, '..', 'venv-deepface', 'Scripts', 'python.exe'), // 👉 use deepface venv
    ['descriptor-matcher-facenet.py', tempFilePath] // 👈 assumes you have a separate deepface descriptor file
  );

  handlePythonResponse(python, tempFilePath, res);
});

// 🔵 Shared handler for Python responses
function handlePythonResponse(pythonProcess, tempFilePath, res) {
  let output = '';

  pythonProcess.stdout.on('data', (data) => {
    output += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error('🐍 Python stderr:', data.toString());
  });

  pythonProcess.on('close', () => {
    fs.unlink(tempFilePath, (err) => {
      if (err) console.error('🧹 Failed to delete temp file:', tempFilePath);
    });

    const lines = output.trim().split('\n');
    const lastJsonLine = lines.reverse().find(line => {
      try {
        JSON.parse(line);
        return true;
      } catch {
        return false;
      }
    });

    if (!lastJsonLine) {
      console.error('❌ No valid JSON in output:\n', output);
      return res.status(200).json([]);
    }

    try {
      const matches = JSON.parse(lastJsonLine);
      return res.status(200).json(matches);
    } catch (e) {
      console.error('❌ Final parse error:', e, '\nRaw line:', lastJsonLine);
      return res.status(200).json([]);
    }
  });
}

app.listen(PORT, () => {
  console.log(`🚀 Server listening on http://localhost:${PORT}`);
});
