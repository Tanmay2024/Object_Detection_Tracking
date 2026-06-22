# 🎥 AI Vision Tracker

## Real-Time Object Detection & Tracking System

AI Vision Tracker is a computer vision application developed using YOLOv8, OpenCV, and Streamlit. The system detects and tracks multiple objects in uploaded videos while displaying bounding boxes, tracking IDs, and live object statistics.

## 🚀 Features

* Real-Time Object Detection
* Multi-Object Tracking
* Unique Tracking IDs
* Video Upload Support
* Live Object Statistics
* YOLOv8 Integration
* Streamlit Interactive Interface
* Vehicle and Person Detection

## 🛠 Technologies Used

* Python
* Streamlit
* OpenCV
* Ultralytics YOLOv8
* NumPy

## 📂 Project Structure

```text
Object_Detection_Tracking/
│
├── app.py
├── requirements.txt
├── README.md
└── sample_video.mp4
```

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd Object_Detection_Tracking
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🎯 How It Works

1. Upload a video file.
2. YOLOv8 processes each frame.
3. Objects are detected and classified.
4. Tracking IDs are assigned to detected objects.
5. Bounding boxes and labels are displayed.
6. Live object statistics are shown.

## 🔍 Detectable Objects

* Person
* Car
* Bus
* Motorcycle
* Truck
* Bicycle
* Other COCO dataset objects

## 📸 Output

The application displays:

* Object Bounding Boxes
* Tracking IDs
* Confidence Scores
* Real-Time Statistics

## 👨‍💻 Author

**Tanmay Guruvugari**

CodeAlpha Artificial Intelligence Internship 2026
