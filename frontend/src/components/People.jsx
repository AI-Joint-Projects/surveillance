import React, { useState, useEffect } from "react";
import axios from "axios";
import "../People.css";

const People = () => {
  const [people, setPeople] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    dob: "",
    role: "",
    personId: "",
    phoneNumber: "",
    images: [],
  });
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPeople = async () => {
      try {
        const response = await axios.get(
          "http://localhost:8000/people/list"
        );
        if (Array.isArray(response.data.people)) {
          setPeople(response.data.people);
        } else {
          console.error("Unexpected API response:", response.data);
          setPeople([]);
        }
      } catch (err) {
        console.error("Error fetching people:", err);
        setError("Failed to load people. Please try again later.");
      }
    };
    fetchPeople();
  }, []);

  const handleDelete = async (personId) => {
    try {
      await axios.delete(
        `http://localhost:8000/people/remove?person_id=${personId}`
      );
      setPeople(people.filter((person) => person.person_id !== personId));
      setSuccessMessage("Person removed successfully!");
      setTimeout(() => setSuccessMessage(""), 5000);
    } catch (error) {
      console.error("Error deleting person:", error);
      setError("Failed to delete person. Please try again later.");
      setTimeout(() => setError(""), 5000);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleImageChange = (e) => {
    const files = Array.from(e.target.files);
    setFormData({ ...formData, images: files });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const form = new FormData();
    form.append("first_name", formData.firstName);
    form.append("last_name", formData.lastName);
    form.append("birth_date", new Date(formData.dob).toISOString());
    form.append("role", formData.role);
    form.append("person_id", formData.personId);
    form.append("phone_number", formData.phoneNumber);
    formData.images.forEach((image) => form.append("images", image));

    try {
      const response = await axios.post(
        "http://localhost:8000/people/register",
        form,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      const newPerson = response.data.person_data;
      setPeople((prevPeople) => [...prevPeople, newPerson]);
      setShowModal(false);
      setFormData({
        firstName: "",
        lastName: "",
        dob: "",
        role: "",
        personId: "",
        phoneNumber: "",
        images: [],
      });
      setSuccessMessage("Person registered successfully!");
      setTimeout(() => setSuccessMessage(""), 5000);
    } catch (err) {
      console.error("Error registering person:", err);
      setError("Failed to register person. Please try again later.");
      setTimeout(() => setError(""), 5000);
    } finally {
      setLoading(false);
    }
  };

  const handleAddPerson = () => {
    setShowModal(true);
  };

  return (
    <div className="people-container">
      {/* Success Message */}
      {successMessage && <p className="success-message">{successMessage}</p>}

      {/* Error Message */}
      {error && <p className="error-message">{error}</p>}

      {/* Header Section */}
      <div className="people-header">
        <div className="search-bar">
          <input type="text" placeholder="Search people..." />
          <span className="search-icon">🔍</span>
        </div>
        <button className="add-button" onClick={handleAddPerson}>
          Add People +
        </button>
      </div>

      {/* Cards Section */}
      <div className="people-list">
        {Array.isArray(people) && people.length > 0 ? (
          people.map((person) => (
            <div key={person.person_id} className="people-card">
              <div className="avatar-section">
                {Array.isArray(person.images) &&
                  person.images.map((image, index) => (
                    <img
                      key={index}
                      src={image}
                      alt={`${person.first_name} avatar`}
                      className="avatar"
                    />
                  ))}
              </div>
              <div>
                <h3>{`${person.first_name} ${person.last_name}`}</h3>
                <p>{`DOB: ${person.birth_date}, Role: ${person.role}`}</p>
              </div>
              <button
                className="delete-button"
                onClick={() => handleDelete(person.person_id)}
              >
                Delete
              </button>
            </div>
          ))
        ) : (
          <p>No people found.</p>
        )}
      </div>

      {/* Modal for Adding New Person */}
      {showModal && (
        <div className="modal">
          <div className="modal-content">
            <h3>Add New Person</h3>
            <form onSubmit={handleSubmit}>
              <input
                type="text"
                name="firstName"
                placeholder="First Name"
                value={formData.firstName}
                onChange={handleInputChange}
                required
              />
              <input
                type="text"
                name="lastName"
                placeholder="Last Name"
                value={formData.lastName}
                onChange={handleInputChange}
                required
              />
              <input
                type="date"
                name="dob"
                placeholder="Date of Birth"
                value={formData.dob}
                onChange={handleInputChange}
                required
              />
              <input
                type="text"
                name="role"
                placeholder="Role"
                value={formData.role}
                onChange={handleInputChange}
                required
              />
              <input
                type="text"
                name="personId"
                placeholder="Person ID"
                value={formData.personId}
                onChange={handleInputChange}
                required
              />
              <input
                type="text"
                name="phoneNumber"
                placeholder="Phone Number"
                value={formData.phoneNumber}
                onChange={handleInputChange}
                required
              />
              <div className="image-upload">
                <label htmlFor="images">Upload Images:</label>
                <input
                  type="file"
                  id="images"
                  accept="image/*"
                  multiple
                  onChange={handleImageChange}
                />
              </div>
              <div className="modal-actions">
                <button type="submit" className="submit-button" disabled={loading}>
                  {loading ? "Adding..." : "Add"}
                </button>
                <button
                  type="button"
                  className="cancel-button"
                  onClick={() => setShowModal(false)}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default People;
