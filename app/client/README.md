# ✍️ Handwriting Generator Web App

This is a ReactJS-based web app that generates realistic handwriting from user input text using a deep learning model trained on the IAM Online Handwriting dataset.

---

## 🖼️ Demo

🚀 Try the live demo *(coming soon)*

---

## 📦 Tech Stack

### Frontend
- **ReactJS** + **Vite**
- **TailwindCSS** for styling
- **KonvaJS** via `react-konva` for handwriting stroke visualization
- **shadcn/ui** for UI components

### Backend
- **FastAPI** (Python)
- **PyTorch** model (`.pt` file)
- **Docker** container
- Hosted via **Hugging Face Spaces**

---

## 🧠 Model Overview

The model uses a multi-layer LSTM with a soft window attention mechanism, based on the Graves-style handwriting synthesis architecture.

> It learns to generate stroke sequences conditioned on input text, capturing the dynamics of human handwriting.

📓 See the training notebook here: *(coming soon)*
