# Sadak Sahak (सड़क सहक): AI-Powered Advanced Driver Assistance System (ADAS)

[![Intellectual Property India](https://img.shields.io/badge/Intellectual_Property_India-Patent_Pending-orange)](https://ipindiaservices.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-red)](https://www.tensorflow.org/)

**Sadak Sahak** (सड़क सहक) is a next-generation, affordable Advanced Driver Assistance System (ADAS) designed to enhance road safety through pure computer vision. The name is derived from Hindi/Sanskrit: **Sadak** (सड़क) meaning *Road* and **Sahak** (from Sahayak) meaning *Assistant* or *Helper*.

> [!IMPORTANT]
> **Published Indian Patent**: This technology is featured in Indian Patent Application Number **202641002074** (Intellectual Property India). It is engineered as a low-cost solution for vehicle deployment, requiring no external sensors or cloud infrastructure.

---

## 🚀 Key Features

### 1. Unified ADAS Dashboard
A professional Python-based desktop application built with a **Tkinter GUI** that integrates all safety features into a single, real-time command center.
- Real-time visualization of image processing.
- Dynamic feature toggling (Lane Detection, Segmentation, FCW, Blind Spot).
- Screenshot and image navigation system.

### 2. Intelligent Lane Tracking
Utilizes custom-trained **CNN Architectures** to identify and track lane markings with high precision, providing visual overlays to assist the driver in remaining within safe boundaries.

### 3. Semantic Road Segmentation
Powered by specialized **U-Net Architectures** (trained on BDD100K), this module performs pixel-level classification of the entire road environment, distinguishing between road, vehicles, pedestrians, and obstacles.

### 4. Forward Collision Warning (FCW)
A proactive safety layer that monitors a dynamic **Region of Interest (RoI)**. By analyzing the distance and presence of objects directly ahead, the system issues immediate visual warnings to prevent potential collisions.

### 5. Peripheral Awareness (Blind-Spot Monitoring)
A vision-only blind-spot detection system that monitors the lateral edges of the vehicle's perimeter. It identifies approaching or adjacent vehicles without the need for expensive ultrasonic sensors.

---

## 🛠️ Technical Implementation

- **Core Logic**: Deep Learning models utilizing **Fully Convolutional Networks (FCN)** and **U-Net** for semantic masking.
- **Hardware Agnostic**: Designed to run using standard **Camera Input** (RGB) only, making it compatible with existing vehicle hardware and low-cost dashcams.
- **Local Execution**: All processing happens on-device; no data is sent to the cloud, ensuring low latency and high privacy.

---

## 📂 Project Structure

```text
├── assets/                  # Dataset samples and UI assets
├── models/                  # Pre-trained .h5 Deep Learning models
├── utils_scripts/           # Modular drivers for LD, FCW, and Segmentation
├── adas_dashboard.py        # Main GUI Application (Entry Point)
├── index-developed_adas.py  # Backend ADAS Engine
└── README.md                # Project Documentation
```

---

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow / Keras
- OpenCV
- PIL (Pillow)
- Tkinter (standard with Python)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SadakSahayak.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System
To launch the integrated Sadak Sahak Dashboard:
```bash
python adas_dashboard.py
```

---

## 📄 License & Patent
This project represents an affordable deep-learning-based ADAS solution.
- **Patent Application**: 202641002074 (Intellectual Property India)
- **Author**: Pranav Reddy Mamidi, along with teammates and lecturers at Keshav Memorial Institute of Technology, who have helped me present this project.
- 
---
*Empowering every vehicle with intelligent road assistance.*
