import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

st.title("PPE Detection Demo")

# Load YOLO model
model = YOLO(os.getenv('PPE_WEIGHTS_PATH', 'runs/train/best9/weights/best.pt'))

# Email configuration (main area)
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
    with st.spinner("Detecting PPE..."):
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

    alerts = []
    required_ppe = ["Helmet", "Gloves", "Jacket", "Goggles"]
    icon_map = {"Helmet": "🪖", "Gloves": "🧤", "Jacket": "🦺", "Goggles": "🥽"}
    for item in required_ppe:
        if item not in detected_classes:
            alerts.append(f"{icon_map[item]} Missing {item}!")
    if not detected_classes:
        alerts.append("❌ No PPE detected!")

    return image_np, alerts

tab1, tab2 = st.tabs(["Upload Image", "Webcam"])

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], help="Upload a JPG, JPEG, or PNG image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        result_img, alerts = detect_ppe(img_np.copy())
        if alerts:
            st.markdown(f'<div class="alert-box alert-warning">{"<br/>".join(alerts)}</div>', unsafe_allow_html=True)
            if st.button("Send Email Alert", help="Send an email with the detection result and alerts"):
                send_email(
                    "PPE Compliance Alert",
                    "\n".join(alerts),
                    result_img,
                    SENDER_EMAIL,
                    SENDER_PASSWORD,
                    RECEIVER_EMAIL,
                    SMTP_SERVER,
                    SMTP_PORT
                )
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(result_img, caption="Detection Result", use_container_width=True)
            st.download_button("Download Result", data=cv2.imencode('.jpg', result_img)[1].tobytes(), file_name="ppe_result.jpg", mime="image/jpeg", help="Download the detection result image")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    camera_image = st.camera_input("Take a picture", help="Capture a photo using your webcam")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Webcam Image", use_container_width=True)
        img_np = np.array(image)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        result_img, alerts = detect_ppe(img_np.copy())
        if alerts:
            st.markdown(f'<div class="alert-box alert-warning">{"<br/>".join(alerts)}</div>', unsafe_allow_html=True)
            if st.button("Send Email Alert", key="webcam_email", help="Send an email with the detection result and alerts"):
                send_email(
                    "PPE Compliance Alert",
                    "\n".join(alerts),
                    result_img,
                    SENDER_EMAIL,
                    SENDER_PASSWORD,
                    RECEIVER_EMAIL,
                    SMTP_SERVER,
                    SMTP_PORT
                )
        st.image(result_img, caption="Detection Result", use_container_width=True)
        st.download_button("Download Result", data=cv2.imencode('.jpg', result_img)[1].tobytes(), file_name="ppe_result.jpg", mime="image/jpeg", help="Download the detection result image")
    st.markdown('</div>', unsafe_allow_html=True)