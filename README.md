# Face-Scan-Tensorflow-RTX5070Ti
![image](https://github.com/user-attachments/assets/9996c98f-e5b8-489b-b326-77bc8e2d7c1d)

# 🚀 TensorFlow GPU Setup (RTX 5070 Ti) with MTCNN & FaceNet using Docker

This project demonstrates how to run **TensorFlow 2.15.0 on GPU** (compatible with RTX 5070 Ti) inside Docker, along with **MTCNN** for face detection and **keras-facenet** for face recognition.

---

## ✅ System Requirements

- **GPU:** NVIDIA RTX 5070 Ti
- **NVIDIA Driver:** Version 575+
- **CUDA Toolkit:** 12.9
- **cuDNN:** 8.9
- **Docker Image:** `tensorflow/tensorflow:2.15.0-gpu`
- **python 3.9 ++
- **Host OS:** Ubuntu Server 24.04 or equivalent
![image](https://github.com/user-attachments/assets/199b1a52-5e36-4c39-8586-64e1de011a46)

---

## 🧱 Step-by-Step Installation Guide

### 1️⃣ Install NVIDIA Driver, CUDA 12.9, and cuDNN 8.9

Verify GPU and driver:

```bash
nvidia-smi
Expected output includes:

Driver Version: 575.xx

CUDA Version: 12.9

Resources:

NVIDIA Driver

CUDA Toolkit 12.9

cuDNN 8.9

2️⃣ Install Docker and NVIDIA Container Toolkit
bash
Copy
Edit
# Install Docker
sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
3️⃣ Pull TensorFlow GPU Docker Image
bash
Copy
Edit
docker pull tensorflow/tensorflow:2.15.0-gpu
4️⃣ Prepare Your Project Files
Dockerfile
dockerfile
Copy
Edit
FROM tensorflow/tensorflow:2.15.0-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

# Install MTCNN separately
RUN pip install mtcnn

# Set working directory
WORKDIR /app
COPY . /app

# Run script on container start
CMD ["python", "test_gpu.py"]
requirements.txt
txt
Copy
Edit
fastapi
uvicorn
tensorflow==2.15.0
keras-facenet
scipy
test_gpu.py
python
Copy
Edit
from mtcnn import MTCNN
from keras_facenet import FaceNet
import tensorflow as tf

print("✅ TensorFlow version:", tf.__version__)
print("✅ Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
5️⃣ Build and Run the Docker Container
bash
Copy
Edit
# Build the image
docker build -t tf-gpu-facenet .

# Run the container with GPU access
docker run --gpus all --rm -it tf-gpu-facenet
Expected output:

yaml
Copy
Edit
✅ TensorFlow version: 2.15.0
✅ Num GPUs Available: 1
⚙️ Optional: FastAPI Example
Add an API using FastAPI to upload and detect faces.

main.py
python
Copy
Edit
from fastapi import FastAPI, UploadFile
from mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image
import numpy as np

app = FastAPI()
detector = MTCNN()
embedder = FaceNet()

@app.post("/detect")
async def detect(file: UploadFile):
    image = Image.open(file.file).convert('RGB')
    image_np = np.asarray(image)
    results = detector.detect_faces(image_np)
    return { "faces_detected": len(results), "results": results }
To run it inside your container:

dockerfile
Copy
Edit
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
And run with:

bash
Copy
Edit
docker run --gpus all -p 8000:8000 tf-gpu-facenet
🔍 GPU Verification Inside Docker
Run this inside the container:

bash
Copy
Edit
python test_gpu.py
Expected output should show GPU availability:

yaml
Copy
Edit
✅ TensorFlow version: 2.15.0
✅ Num GPUs Available: 1
You can also use:

bash
Copy
Edit
nvidia-smi
📦 Summary of Key Libraries
Library	Purpose
tensorflow	Deep learning framework with GPU
mtcnn	Face detection using MTCNN
keras-facenet	Face embedding using FaceNet
scipy	Cosine similarity & distance metrics
fastapi	RESTful API framework
uvicorn	ASGI server for FastAPI

🧊 Tips
Ensure CUDA and cuDNN versions match the TensorFlow image

Verify GPU visibility with nvidia-smi both on host and inside container

Use nvidia-container-toolkit to enable GPU passthrough in Docker

