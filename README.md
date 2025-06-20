<p align="center">
  <img src="https://github.com/user-attachments/assets/c56a3eba-66c2-436a-ae88-39ad602d76db" width="450" alt="logo"/>
</p>

# 👕 E-PROVA: Intelligent Virtual Try-On System

> This project is a customized adaptation of [IDM-VTON](https://github.com/yisol/IDM-VTON), integrated with a Flask API and notebook-based pipeline to allow seamless virtual try-on experience via Google Colab.

---

## 💡 Project Overview

E-PROVA enables users to upload a person image and a clothing image, and generates a realistic try-on result using a deep learning model.  
The project extends the capabilities of the original IDM-VTON repository by:

- Adding a Flask-based API (`app.py`)
- Automating model download and setup
- Allowing inference via notebooks or API

---

## 📂 Repository Structure

- `E_PROVA.ipynb`  
  → The main Colab notebook for running the project step by step. It includes:
  - Installing dependencies
  - Mounting and loading data from Google Drive
  - Running the Flask app
  - Sending sample images for inference and visualizing the results

- `app.py`  
  → A Flask server that exposes the `/vton` endpoint to receive person & cloth images and return the generated try-on result.

- `tryon.py`  
  → Contains logic to load the trained IDM-VTON model and perform inference.

- `checkpoints/`  
  → Folder containing the pre-trained model weights (automatically downloaded from Google Drive).

- `modified_inference.py`  
  → A modified version of `inference.py` adapted for API integration and callable inference logic.

---

## 🚀 Try It Out (Google Colab)

1. **Open the Colab notebook**  
   [▶️ E_PROVA.ipynb](https://github.com/marwan-shamel1/E-PROVA/blob/main/E_PROVA.ipynb)

2. **Follow the steps in the notebook**:
   - Install required packages
   - Mount Google Drive and load this folder:
     > 📁 [Google Drive Folder](https://drive.google.com/drive/u/1/folders/1ltewajoB8ScpNcTQqFuRXS3-fAEnYBGf)
   - Start the Flask app via `app.py`
   - Send test images via a form or API request and view the try-on results

3. **API Endpoint Example** (via ngrok or local port):
POST /vton
Content-Type: multipart/form-data
Form-data:
- person: [file.jpg]
- cloth: [file.jpg]
Returns:
- image/jpeg: the try-on result image

---

## ⚙️ Requirements

- Python 3.9+
- Flask
- PyTorch
- torchvision
- Pillow
- OpenCV
- Gradio (optional)
- ngrok (for external API exposure)

```bash
pip install -r requirements.txt
```

---

🧠 Model Info
E-PROVA is built on top of IDM-VTON, a diffusion-based virtual try-on model that achieves high realism by preserving garment structure and user pose.
Modifications were made to support API-based interaction and simplified loading for Colab deployment.

---

✅ Features
Ready-to-run notebook interface with no local setup needed

Preloaded checkpoints via Google Drive

Clean and simple REST API using Flask

Can be easily connected to frontends (Gradio, React, etc.)

---
🔗 Useful Links
📁 [Google Drive Folder (Project & Models)](https://drive.google.com/drive/u/1/folders/1ltewajoB8ScpNcTQqFuRXS3-fAEnYBGf)

🧠 [Original IDM-VTON Repo](https://github.com/yisol/IDM-VTON)

🔧 [Backend Integration – E-PROVA GitHub](https://github.com/marwanMagdy66/E-Prova)

