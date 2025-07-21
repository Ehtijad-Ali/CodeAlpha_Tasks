# 🚀 Advanced YOLOv8 Object Tracker GUI

A **real-time**, **high-performance** object detection and tracking application powered by **YOLOv8** and designed with a sleek GUI using **Tkinter**.

This project allows you to detect and track multiple objects live from a webcam or video file, assign them unique IDs, and visualize their movement paths interactively.

> ⚠️ *It is recommended to replace the screenshot below with a GIF of your application in action for better engagement (use tools like ScreenToGif or LICEcap).*

---

## ✨ Features

* **📹 Real-Time Detection & Tracking**
  Detect and track multiple objects simultaneously on live feeds or pre-recorded videos.

* **🎯 Unique Object IDs & Movement Trails**
  Each object is assigned a persistent ID with a visual trail that follows its path frame by frame.

* **🖥️ Modern & Interactive UI**
  A responsive dark-themed interface created with Tkinter and `ttkthemes` for a seamless user experience.

* **🎛️ Real-Time Control Panel**

  * Adjust confidence threshold using a slider
  * Toggle bounding boxes and movement trails
  * Select YOLOv8 model variants (n, s, m)

* **📊 Live Statistics Display**

  * Real-time FPS counter
  * Live count of currently tracked objects

* **🎥 Flexible Video Source Options**

  * Use your system’s webcam
  * Load and process local video files (.mp4, .avi, etc.)

* **⏯️ Playback Control**

  * Pause, resume, and stop video playback
  * Real-time progress bar for videos

---

## 🛠️ Tech Stack

This project is built with the following tools and libraries:

* [Python 3.8+](https://www.python.org/)
* [Tkinter](https://docs.python.org/3/library/tkinter.html)
* [ttkthemes](https://pypi.org/project/ttkthemes/)
* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [Torch](https://pytorch.org/)
* [Pillow](https://python-pillow.org/)

---

## ⚙️ Getting Started

### Prerequisites

* Python 3.8 or newer
* An IDE (e.g., VS Code, PyCharm) or any Python-compatible editor

### Installation Steps

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **Create a Virtual Environment (Recommended)**

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Required Libraries**

```bash
pip install opencv-python ultralytics torch torchvision pillow ttkthemes
```

4. **Download a YOLOv8 Model**

Download `yolov8n.pt` or another model from the [Ultralytics YOLOv8 release page](https://github.com/ultralytics/ultralytics). Place the `.pt` file in your project directory.

---

## ▶️ Running the Application

After completing setup, launch the application by running:

```bash
python your_script_name.py
```

Replace `your_script_name.py` with the actual filename of your Python script.

---

## 📖 How to Use

1. **Select a YOLO Model**
   Use the dropdown to choose between models like `yolov8n`, `yolov8s`, or `yolov8m`.

2. **Choose a Video Source**

   * Click **Start Webcam** to use the camera.
   * Click **Open Video File** to load a video from your system.

3. **Customize Settings in Real Time**

   * Adjust detection confidence via slider.
   * Toggle bounding boxes or object trails.

4. **Playback Controls**

   * Pause, resume, or stop video streams.
   * Follow video progress via the bottom progress bar.

---

## 📁 Project Structure

```
.
├── your_script_name.py       # Main application script
├── yolov8n.pt                # YOLOv8 model file (or yolov8s.pt, yolov8m.pt, etc.)
└── README.md                 # Project documentation
```

