import React, { useState } from "react";
import "../Survillence.css";

const Surveillance = () => {
  const [startDatetime, setStartDatetime] = useState("");
  const [endDatetime, setEndDatetime] = useState("");
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");
  const [modalImage, setModalImage] = useState(null); // To handle the image preview modal

  const handleConfirm = async () => {
    if (!startDatetime || !endDatetime) {
      alert("Please select both start and end datetime.");
      return;
    }

    try {
      console.log("Fetching data for range:", { startDatetime, endDatetime });

      const response = await fetch(
        `http://localhost:8000/surveillance/search?start_time=${encodeURIComponent(
          startDatetime
        )}&end_time=${encodeURIComponent(endDatetime)}`
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to fetch data.");
      }

      const data = await response.json();
      console.log("Received data:", data);

      setResults(data.data || []);
      setError("");
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(err.message);
      setResults([]);
    }
  };

  const handleImageClick = (imgUrl) => {
    setModalImage(imgUrl); // Show the clicked image in the modal
  };

  const closeModal = () => {
    setModalImage(null); // Close the modal
  };

  return (
    <div className="surveillance-container">
      <h1 className="section-title">Surveillance</h1>
      <p className="intro-text">
        Enter a datetime range and click confirm to view detected people.
      </p>
      <div className="time-range-container">
        <p className="sub-title">Datetime Range Selection</p>
        <div className="date-picker-section">
          <label className="date-picker-label" htmlFor="start-datetime">
            Start Datetime:
          </label>
          <input
            type="datetime-local"
            id="start-datetime"
            className="date-picker"
            value={startDatetime}
            onChange={(e) => setStartDatetime(e.target.value)}
          />
        </div>
        <div className="date-picker-section">
          <label className="date-picker-label" htmlFor="end-datetime">
            End Datetime:
          </label>
          <input
            type="datetime-local"
            id="end-datetime"
            className="date-picker"
            value={endDatetime}
            onChange={(e) => setEndDatetime(e.target.value)}
          />
        </div>
        <button className="confirm-button" onClick={handleConfirm}>
          Confirm Selection
        </button>
      </div>

      {/* Display results */}
      <div className="results-section">
        {error && <p className="error-message">{error}</p>}
        {results.length > 0 ? (
          <ul className="results-list">
            {results.map((result, index) => (
              <li key={index} className="result-item">
                <strong>Person ID:</strong> {result.person_id || "None"} <br />
                <strong>Time Entered:</strong>{" "}
                {result.time_entered
                  ? new Date(result.time_entered).toLocaleString()
                  : "None"}{" "}
                <br />
                <strong>Time Exited:</strong>{" "}
                {result.time_exited
                  ? new Date(result.time_exited).toLocaleString()
                  : "None"}{" "}
                <br />
                {result.frame_images_urls &&
                  result.frame_images_urls.length > 0 && (
                    <div>
                      <p>
                        <strong>Frame Images:</strong>
                      </p>
                      <div className="inline-images-row">
                        {result.frame_images_urls.map((imgUrl, imgIndex) => (
                          <button
                            key={imgIndex}
                            className="image-button"
                            onClick={() => handleImageClick(imgUrl)}
                          >
                            <img
                              src={imgUrl}
                              alt={`Frame ${imgIndex}`}
                              className="frame-image-inline"
                            />
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
              </li>
            ))}
          </ul>
        ) : (
          !error && (
            <p className="no-data-message">
              No data found for the selected time range.
            </p>
          )
        )}
      </div>

      {/* Modal for image preview */}
      {modalImage && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content">
            <img src={modalImage} alt="Full Preview" className="modal-image" />
          </div>
        </div>
      )}
    </div>
  );
};

export default Surveillance;
