
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import mediapipe as mp
import os
from dotenv import load_dotenv
load_dotenv()
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

st.title("PPE Detection Demo")

# Load YOLO model once
model = YOLO(os.getenv('PPE_WEIGHTS_PATH', 'runs/train/best9/weights/best.pt'))

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5,  # Lowered from 0.7
    min_tracking_confidence=0.5
)


# Email configuration UI
with st.sidebar:
    st.header("Email Settings")
    SENDER_EMAIL = st.text_input("Sender Email", value=os.getenv("SENDER_EMAIL", ""))
    SENDER_PASSWORD = st.text_input("Sender Password", type="password", value=os.getenv("SENDER_PASSWORD", ""))
    RECEIVER_EMAIL = st.text_input("Receiver Email", value=os.getenv("RECEIVER_EMAIL", ""))
    SMTP_SERVER = st.text_input("SMTP Server", value=os.getenv("SMTP_SERVER", "smtp.gmail.com"))
    SMTP_PORT = st.number_input("SMTP Port", value=int(os.getenv("SMTP_PORT", "587")), step=1)

def send_email(subject, body, image_np, sender_email, sender_password, receiver_email, smtp_server, smtp_port):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg.attach(MIMEText(f"{body}\n\nTimestamp: {timestamp}", 'plain'))
        # Save image to buffer
        _, img_encoded = cv2.imencode('.jpg', image_np)
        msg.attach(MIMEImage(img_encoded.tobytes(), name="alert.jpg"))
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Email failed: {e}")

def detect_ppe(image_np):
    # YOLO detection
    results = model.predict(source=image_np, imgsz=640, conf=0.4, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_names = ["Shoes", "Goggles", "Gloves", "Helmet", "Jacket"]
    detected_classes = set()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(class_ids[i])
        detected_classes.add(class_names[class_id])
        label = f"{class_names[class_id]} {confidences[i]:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Pose detection
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    alerts = []
    current_time = datetime.now()
    def get_torso_center(landmarks, image_shape):
        try:
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            mid_shoulder = ((left_shoulder.x + right_shoulder.x)/2 * image_shape[1],
                            (left_shoulder.y + right_shoulder.y)/2 * image_shape[0])
            mid_hip = ((left_hip.x + right_hip.x)/2 * image_shape[1],
                       (left_hip.y + right_hip.y)/2 * image_shape[0])
            return ((mid_shoulder[0] + mid_hip[0])/2, (mid_shoulder[1] + mid_hip[1])/2)
        except:
            return None
    def get_eye_centers(landmarks, image_shape):
        try:
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
            return (
                (left_eye.x * image_shape[1], left_eye.y * image_shape[0]),
                (right_eye.x * image_shape[1], right_eye.y * image_shape[0])
            )
        except:
            return None, None
    def is_near(object_box, keypoint, threshold=100):
        x1, y1, x2, y2 = object_box
        kp_x, kp_y = keypoint
        return x1 - threshold < kp_x < x2 + threshold and y1 - threshold < kp_y < y2 + threshold
    def is_glove_properly_worn(detection, landmarks, image_shape):
        try:
            left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image_shape[0])
            right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_shape[1],
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_shape[0])
            left_hand = is_near(detection, left_wrist, 80)
            right_hand = is_near(detection, right_wrist, 80)
            return left_hand or right_hand
        except:
            return False

    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image_np, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark
        head_kp = (landmarks[mp_pose.PoseLandmark.NOSE].x * image_np.shape[1],
                   landmarks[mp_pose.PoseLandmark.NOSE].y * image_np.shape[0])
        torso_center = get_torso_center(landmarks, image_np.shape)
        left_eye, right_eye = get_eye_centers(landmarks, image_np.shape)
        required_ppe = ["Helmet", "Gloves", "Jacket", "Goggles"]
        ppe_status = {item: {"detected": False, "properly_worn": False} for item in required_ppe}
        for item in required_ppe:
            if item in detected_classes:
                ppe_status[item]["detected"] = True
        for i, detection in enumerate(boxes):
            class_id = int(class_ids[i])
            item_name = class_names[class_id]
            confidence = confidences[i]
            if confidence < 0.5:
                continue
            if item_name == "Helmet" and head_kp:
                ppe_status["Helmet"]["properly_worn"] = is_near(detection, head_kp, 100)
            elif item_name == "Gloves":
                ppe_status["Gloves"]["properly_worn"] = is_glove_properly_worn(detection, landmarks, image_np.shape)
            elif item_name == "Jacket" and torso_center:
                ppe_status["Jacket"]["properly_worn"] = is_near(detection, torso_center, 200)
            elif item_name == "Goggles" and left_eye and right_eye:
                ppe_status["Goggles"]["properly_worn"] = (
                    is_near(detection, left_eye, 50) or is_near(detection, right_eye, 50)
                )
        # If all PPE is detected and properly worn, show a positive message
        if all(ppe_status[item]["detected"] and ppe_status[item]["properly_worn"] for item in required_ppe):
            alerts.append("All PPE items are detected and properly worn!")
        else:
            for item in required_ppe:
                if not ppe_status[item]["detected"]:
                    alerts.append(f"Missing {item}!")
                elif not ppe_status[item]["properly_worn"]:
                    alerts.append(f"{item} not properly worn!")
        # Visualize wrist landmarks
        try:
            left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_np.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image_np.shape[0])
            right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_np.shape[1],
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_np.shape[0])
            cv2.circle(image_np, (int(left_wrist[0]), int(left_wrist[1])), 8, (255, 0, 0), -1)
            cv2.circle(image_np, (int(right_wrist[0]), int(right_wrist[1])), 8, (255, 0, 0), -1)
        except:
            pass
    else:
        if len(boxes) > 0:
            alerts.append("PPE detected, but pose landmarks not found. Please try another image or check image quality.")
        else:
            alerts.append("No person detected!")

    # Show alerts
    if alerts:
        st.warning("\n".join(alerts))
        if st.button("Send Email Alert"):
            send_email(
                "PPE Compliance Alert",
                "\n".join(alerts),
                image_np,
                SENDER_EMAIL,
                SENDER_PASSWORD,
                RECEIVER_EMAIL,
                SMTP_SERVER,
                SMTP_PORT
            )
    return image_np

tab1, tab2 = st.tabs(["Upload Image", "Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        result_img = detect_ppe(img_np.copy())
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(result_img, caption="Detection Result", use_container_width=True)

with tab2:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Webcam Image", use_container_width=True)
        img_np = np.array(image)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        result_img = detect_ppe(img_np.copy())
        st.image(result_img, caption="Detection Result", use_container_width=True)
