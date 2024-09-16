# Image Classification with FastAPI and Streamlit

This repository contains an image classification system built using PyTorch, FastAPI for the backend, and Streamlit for the frontend. The model classifies images into four categories: **Smoke**, **Fire**, **None**, and **Smoke and Fire**.

## Features
- **PyTorch Model**: Built with EfficientNet-B1 to classify images.
- **FastAPI Backend**: A REST API for serving the model and making predictions.
- **Streamlit Frontend**: A user-friendly web interface for uploading images and receiving classification results.
- **Multi-image Upload**: Allows users to upload and classify multiple images at once.

## Requirements
- **Python 3.8+**
- **Dependencies**:
  - PyTorch
  - FastAPI
  - Uvicorn
  - Streamlit
  - PIL (Python Imaging Library)
  - Requests

## Installation

### Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### Set up the virtual environment:
It is recommended to use `venv` or `conda` to create an isolated environment.

#### Using `venv`:
```bash
python -m venv torch-env
source torch-env/bin/activate  # On Windows: torch-env\Scripts\activate
```

#### Using `conda`:
```bash
conda create --name torch-env python=3.8
conda activate torch-env
```

### Install the required dependencies:
```bash
pip install -r requirements.txt
```

If you need to manually install the dependencies:
```bash
pip install torch torchvision fastapi uvicorn pillow streamlit requests
```

## Model Preparation
Ensure your PyTorch model is saved as `final_model.pth` in the root directory of the repository. If you need to retrain or update the model, ensure it is saved with this name.

## Directory Structure
```
.
├── app.py               # FastAPI backend
├── streamlit_app.py      # Streamlit frontend
├── final_model.pth       # PyTorch model file (ensure you add your own)
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Running the Application

### 1. Start the FastAPI Backend
The FastAPI server provides the backend for image classification. Start it using Uvicorn:
```bash
uvicorn app:app --reload
```
This will start the FastAPI server at `http://127.0.0.1:8000`.

### 2. Run the Streamlit Frontend
In a new terminal window (while the FastAPI server is running), start the Streamlit app:
```bash
streamlit run streamlit_app.py
```
This will launch the frontend interface where you can upload images and get classification predictions.

### 3. Access the Application
- FastAPI server: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Streamlit interface: [http://localhost:8501](http://localhost:8501)

## Usage
1. **Upload Images**: In the Streamlit interface, click the "Browse files" button to upload one or more images.
2. **Classify Images**: After uploading, click the "Classify Images" button to send the images to the FastAPI backend.
3. **View Results**: The classification results will be displayed along with the uploaded images.

## API Documentation
Once the FastAPI server is running, you can view the interactive API documentation at:
```bash
http://127.0.0.1:8000/docs
```

### Example Request
You can send a POST request using `curl` or any other HTTP client to test the FastAPI API:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -F 'file=@your_image.png'
```

## Troubleshooting
- Ensure both FastAPI and Streamlit are running in the same environment.
- If you encounter CORS issues, you may need to add appropriate CORS settings in the FastAPI app.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
