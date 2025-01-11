import cv2
import numpy as np
import streamlit as st
import tempfile
import os

# Define body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Model configuration
MODEL_PATH = r"D:\project\human poistore\graph_opt.pb"  # Replace with the correct model file path
inWidth, inHeight = 368, 368
threshold = 0.2

@st.cache_resource
def load_model(model_path):
    """
    Load the TensorFlow model for pose estimation.
    """
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None
    try:
        net = cv2.dnn.readNetFromTensorflow(model_path)
        return net
    except cv2.error as e:
        st.error(f"Error loading model: {e}")
        return None

def estimate_pose(frame, net, threshold):
    """
    Estimate poses on a single frame.
    """
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]

    # Prepare the input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform inference
    out = net.forward()
    out = out[:, :len(BODY_PARTS), :, :]  # Restrict to body parts

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    return points

def draw_pose(frame, points):
    """
    Draw pose skeleton on the frame.
    """
    for pair in POSE_PAIRS:
        partFrom, partTo = pair[0], pair[1]
        if partFrom in BODY_PARTS and partTo in BODY_PARTS:
            idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (5, 5), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (5, 5), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Streamlit UI
st.set_page_config(page_title="Pose Estimation App", page_icon="🎥", layout="wide")

st.title("🎥 Pose Estimation in Video")
st.markdown("Upload a video file below to perform real-time pose estimation using a pre-trained model.")

uploaded_video = st.file_uploader("Upload a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

# Load model
net = load_model(MODEL_PATH)

if uploaded_video and net:
    st.sidebar.success("Model loaded successfully.")
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open the uploaded video file.")
    else:
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 360))

            # Estimate pose and draw it on the frame
            points = estimate_pose(frame, net, threshold)
            output_frame = draw_pose(frame, points)

            # Convert frame to RGB for Streamlit display
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            # Display frame
            stframe.image(output_frame, channels="RGB", use_column_width=True)

        cap.release()
        st.success("Video processing complete!")
else:
    if not net:
        st.info("Waiting for the model to load. Please check the model path.")
    else:
        st.info("Please upload a video file to begin processing.")
