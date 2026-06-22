import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

st.set_page_config(
    page_title="AI Vision Tracker",
    page_icon="🎥",
    layout="wide"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
    background:#0a0f0a;
    color:white;
}

.main-title{
    text-align:center;
    color:#00ff88;
    font-size:55px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#a7f3d0;
    margin-bottom:20px;
}

.stats{
    background:#111827;
    border:1px solid #00ff88;
    border-radius:15px;
    padding:10px;
    text-align:center;
    margin-bottom:15px;
}

.footer{
    text-align:center;
    color:#9ca3af;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
"""
<div class='main-title'>
🎥 AI Vision Tracker
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='sub-title'>
Real-Time Object Detection & Tracking System
</div>
""",
unsafe_allow_html=True
)

# ---------------- VIDEO UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "📹 Upload Video",
    type=["mp4", "avi", "mov"]
)

# ---------------- PROCESS ---------------- #

if uploaded_file:

    temp_file = tempfile.NamedTemporaryFile(
        delete=False
    )

    temp_file.write(uploaded_file.read())

    model = YOLO("yolov8n.pt")

    st.success(
        "🟢 AI Engine Online - Processing Video Feed"
    )

    cap = cv2.VideoCapture(
        temp_file.name
    )

    frame_placeholder = st.empty()

    stats_placeholder = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
                  frame,
                  persist=True,
                  conf=0.5,
                  verbose=False
                  )

        annotated_frame = results[0].plot(
    line_width=1
        )

        person_count = 0
        car_count = 0
        motorcycle_count = 0
        bus_count = 0

        if results[0].boxes is not None:

            for cls in results[0].boxes.cls:

                class_name = model.names[int(cls)]

                if class_name == "person":
                    person_count += 1

                elif class_name == "car":
                    car_count += 1

                elif class_name == "motorcycle":
                    motorcycle_count += 1

                elif class_name == "bus":
                    bus_count += 1

        stats_placeholder.markdown(
            f"""
            <div class='stats'>
            👥 Persons: <b>{person_count}</b>
            &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
            🚗 Cars: <b>{car_count}</b>
            &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
            🏍 Motorcycles: <b>{motorcycle_count}</b>
            &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
            🚌 Buses: <b>{bus_count}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        annotated_frame = results[0].plot()

        frame_placeholder.image(
            annotated_frame,
            channels="BGR",
            use_container_width=True
        )

    cap.release()

    st.success(
        "✅ Analysis Complete - All Objects Tracked Successfully"
    )

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.markdown("""
<div class='footer'>
Developed by <b>Tanmay Guruvugari</b><br>
CodeAlpha Artificial Intelligence Internship 2026
</div>
""", unsafe_allow_html=True)

