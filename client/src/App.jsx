import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const videoRef = useRef();
  const canvasRef = useRef();
  const intervalRef = useRef(null);
  const isPausedRef = useRef(false);
  const isFetchingRef = useRef(false);

  const [faceapiMatches, setFaceapiMatches] = useState([]);
  const [insightMatches, setInsightMatches] = useState([]);
  const [deepfaceMatches, setDeepfaceMatches] = useState([]);
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    const setupCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    };

    setupCamera();
  }, []);

  useEffect(() => {
    isPausedRef.current = isPaused;

    if (intervalRef.current) clearInterval(intervalRef.current);

    if (!isPaused) {
      const startFetching = async () => {
        if (!canvasRef.current || !videoRef.current || isPausedRef.current) {
          return;
        }

        if (isFetchingRef.current) {
          return; // already fetching, wait
        }

        isFetchingRef.current = true;

        const ctx = canvasRef.current.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, 320, 240);
        const dataUrl = canvasRef.current.toDataURL('image/png');

        try {
          const [faceapiRes, insightRes, deepfaceRes] = await Promise.all([
            axios.post('http://localhost:5000/get-matches', { imageData: dataUrl }),
            axios.post('http://localhost:5000/get-matches-insight', { imageData: dataUrl }),
            axios.post('http://localhost:5000/get-matches-deepface', { imageData: dataUrl }),
          ]);

          if (!isPausedRef.current) {
            if (faceapiRes.data && faceapiRes.data.length > 0) {
              const urls = faceapiRes.data.map((match) => `http://localhost:5000/${match.file}`);
              setFaceapiMatches(urls);
            }

            if (insightRes.data && insightRes.data.length > 0) {
              const urls = insightRes.data.map((match) => `http://localhost:5000/${match.file}`);
              setInsightMatches(urls);
            }

            if (deepfaceRes.data && deepfaceRes.data.length > 0) {
              const urls = deepfaceRes.data.map((match) => `http://localhost:5000/${match.file}`);
              setDeepfaceMatches(urls);
            }
          }
        } catch (error) {
          console.error('❌ Error during fetch:', error.message);
        }

        isFetchingRef.current = false;

        // Start next fetch after a delay
        setTimeout(startFetching, 5000);
      };

      startFetching();
    }

    return () => clearInterval(intervalRef.current);
  }, [isPaused]);

  const togglePause = () => {
    setIsPaused((prev) => !prev);
  };

  return (
    <div className="container">
      <div className="camera-wrapper">
        <video ref={videoRef} autoPlay width="640" height="480" className="mirrored-video" />
        <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
      </div>

      <div className="button-wrapper">
        <button className="pause-button" onClick={togglePause}>
          {isPaused ? 'Resume' : 'Pause'}
        </button>
      </div>

      <h2>FaceAPI Matches</h2>
      <div className="thumbnail-row">
        {Array.from({ length: 26 }).map((_, i) => (
          <div key={i} className="thumbnail-slot">
            {faceapiMatches[i] && (
              <img
                src={faceapiMatches[i]}
                alt={`faceapi-match-${i}`}
                className="thumbnail-img"
              />
            )}
          </div>
        ))}
      </div>

      <h2>InsightFace Matches</h2>
      <div className="thumbnail-row">
        {Array.from({ length: 26 }).map((_, i) => (
          <div key={i} className="thumbnail-slot">
            {insightMatches[i] && (
              <img
                src={insightMatches[i]}
                alt={`insight-match-${i}`}
                className="thumbnail-img"
              />
            )}
          </div>
        ))}
      </div>

      <h2>DeepFace Matches</h2>
      <div className="thumbnail-row">
        {Array.from({ length: 26 }).map((_, i) => (
          <div key={i} className="thumbnail-slot">
            {deepfaceMatches[i] && (
              <img
                src={deepfaceMatches[i]}
                alt={`deepface-match-${i}`}
                className="thumbnail-img"
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
