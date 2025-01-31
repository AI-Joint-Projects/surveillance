import React, { useState } from "react";
import Header from "./components/Home";
import People from "./components/People";
import './Header.css';
import Surveillance from "./components/Survillence";
import Webcam from "./Webcam";

const App = () => {
  const [currentPage, setCurrentPage] = useState("Home");

  const handleButtonClick = (page) => {
    setCurrentPage(page); // Update the current page
  };

  return (
    <div className="header-container">
      <div className="sidebar">
        <div className="admin-section">
          <div className="admin-icon"></div>
          <span className="admin-text">Admin</span>
        </div>
        <button
          onClick={() => handleButtonClick('Home')}
          className={`sidebar-button ${currentPage === "Home" ? "active" : ""}`}
        >
          Home
        </button>
        <button
          onClick={() => handleButtonClick('People')}
          className={`sidebar-button ${currentPage === "People" ? "active" : ""}`}
        >
          People
        </button>

        <button
          onClick={() => handleButtonClick('Surveillance')}
          className={`sidebar-button ${currentPage === "Surveillance" ? "active" : ""}`}
        >
          Surveillance
        </button>
      </div>

      <div className="content">
        {currentPage === "Home" && <Header />}
        {currentPage === "People" && <People />}
        {currentPage === "Webcam" && <Webcam/>}
        {currentPage === "Surveillance" && <Surveillance/>}
      </div>
    </div>
  );
};

export default App;
