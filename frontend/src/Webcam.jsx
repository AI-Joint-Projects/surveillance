import React, { useEffect, useRef, useState } from "react";
import "./Webcam.css";

const Webcam = () => {
  const videoRef = useRef(null);
  const [hasPermission, setHasPermission] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: true,
          audio: false 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setHasPermission(true);
        }
      } catch (err) {
        setError("Failed to access webcam. Please make sure you have granted permission.");
        console.error("Error accessing webcam:", err);
      }
    };

    startWebcam();

    // Cleanup function to stop the webcam when component unmounts
    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="webcam-container">
      <h1 className="welcome-text">Welcome to Webcam</h1>
      {error ? (
        <div className="error-message">{error}</div>
      ) : (
        <video 
          ref={videoRef}
          autoPlay
          playsInline
          className="webcam-video"
        />
      )}
    </div>
  );
};

export default Webcam;