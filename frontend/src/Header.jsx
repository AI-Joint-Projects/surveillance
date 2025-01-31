import React from 'react';
import './Header.css';

const Header = () => {
  const handleButtonClick = (feature) => {
    alert(`${feature} clicked!`);
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
          className="sidebar-button"
        >
          Home
        </button>
        <button
          onClick={() => handleButtonClick('People')}
          className="sidebar-button"
        >
          People
        </button>
        <button
          onClick={() => handleButtonClick('Webcam')}
          className="sidebar-button"
        >
          Webcam
        </button>
        <button
          onClick={() => handleButtonClick('Surveillance')}
          className="sidebar-button"
        >
          Surveillance
        </button>
      </div>
      <div className="main-content">
        <div className="content-wrapper">
          <div className="text-section">
            <h1>Welcome to the Dashboard</h1>
            <p>Select an option from the sidebar to proceed.</p>
          </div>
          
        </div>
      </div>
    </div>
  );
};

export default Header;