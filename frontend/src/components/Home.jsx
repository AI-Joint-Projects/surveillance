import React from 'react';
import '../Header.css';
import dashboardImage from '../assets/AdobeStock_204895312.jpeg'; // Adjust the path to your image

const Header = () => {
  const handleButtonClick = (feature) => {
    alert(`${feature} clicked!`);
  };

  return (
    <div className="main-content">
      <h1>Welcome to the Dashboard</h1>
      <p>Select an option from the sidebar to proceed.</p>
      <img src={dashboardImage} alt="Dashboard" className="dashboard-image" />
    </div>
  );
};

export default Header;