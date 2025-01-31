# Open Set Face Recognition and Surveillance


## Getting Started

### Prerequisites
Ensure you have the following installed:
- Node.js (for the frontend)
- Python (for the backend)
- MongoDB (for metadata storage)
- Qdrant (for vector embeddings storage)
- Pytorch
- Cuda v>=1.8

## Model 
```
cd model
python model.py
```

## Surveillance App
This is a surveillance application that leverages open-set face recognition to identify individuals in real time. It consists of a frontend and a backend working together to process and recognize faces from a live video feed.



### Installation & Setup

#### Frontend
```
cd frontend
npm install
npm run dev
```

#### Backend
```
cd backend
```

Download the model file (best.pt) and (epoch_43) from the given link and place it in the app folder.
link: https://drive.google.com/drive/folders/1Sha9HVwtJ1PO84hd593YpDnROO2SPplB

##### Install the required dependencies:

```
pip install -r requirements.txt
```
##### Configure the database:
Set up MongoDB and Qdrant.
Update database.py with the database connection details.

##### Run the backend server:

```
uvicorn app.main:app --reload

```


#### Features
Real-time Face Recognition: Detects and recognizes faces from a video feed.
Open-Set Recognition: Identifies known and unknown individuals dynamically.
Scalable Architecture: Uses Qdrant for embedding storage and MongoDB for metadata.
#### Contributing
Feel free to contribute by submitting issues or pull requests.

#### License
This project is open-source under the MIT License.






