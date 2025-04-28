// server/index.js
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json({ limit: '10mb' })); // Support large base64 images

app.post('/snapshot', (req, res) => {
  const { imageData } = req.body;

  if (!imageData) {
    return res.status(400).send('No image data received.');
  }

  console.log('Received snapshot (base64 length):', imageData.length);
  // Do nothing for now
  res.sendStatus(200);
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
