
import os
from dotenv import load_dotenv
load_dotenv()
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision
import mediapipe as mp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
import threading

# ====== Global Variables for Cooldown ======
LAST_EMAIL_SENT = datetime.min
EMAIL_COOLDOWN = timedelta(seconds=10)  
ALERT_HISTORY = {}
TERMINAL_COOLDOWN = timedelta(seconds=10)
LAST_TERMINAL_OUTPUT = datetime.min
TERMINAL_SPEED_COOLDOWN = timedelta(seconds=10)
LAST_SPEED_OUTPUT = datetime.min

def _add_alert(alerts, message, current_time):
    """Helper function to manage alert cooldowns."""
    global ALERT_HISTORY
    if message not in ALERT_HISTORY or (current_time - ALERT_HISTORY[message]).total_seconds() > 60:
        alerts.append(message)
        ALERT_HISTORY[message] = current_time

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 model via relative path or env var
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), 'runs', 'train', 'best9', 'weights', 'best.pt')
WEIGHTS_PATH = os.getenv('PPE_WEIGHTS_PATH', DEFAULT_WEIGHTS)
yolo_model = YOLO(WEIGHTS_PATH).to(device)
print("Welcome")

# YOLO Warmup
dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
yolo_model.predict(source=dummy_image, imgsz=640, verbose=False)
print("YOLO warmup completed")

# Mask R-CNN setup
class_names = ["Shoes", "Goggles", "Gloves", "Helmet", "Jacket"]
num_classes = len(class_names) + 1

def get_pretrained_mask_rcnn(num_classes=91):
    print("Loading Mask R-CNN model...")
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        if num_classes != 91:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                in_features_mask, 256, num_classes
            )
            print("Mask R-CNN model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Mask R-CNN: {e}")
        return None

mask_rcnn_model = get_pretrained_mask_rcnn(num_classes)
if mask_rcnn_model:
    mask_rcnn_model.to(device).eval()
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    with torch.no_grad():
        mask_rcnn_model(dummy_input)
    print("Mask R-CNN warmup completed")

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Email configuration (use environment variables)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", SENDER_EMAIL)

def send_email_async(subject, body, image_path):
    if datetime.now() - LAST_EMAIL_SENT < EMAIL_COOLDOWN:
        cooldown_left = (EMAIL_COOLDOWN - (datetime.now() - LAST_EMAIL_SENT)).seconds
        print(f"⏳ Email cooldown active ({cooldown_left}s remaining) - skipping send")
        return
    email_thread = threading.Thread(target=send_email, args=(subject, body, image_path))
    email_thread.start()

def send_email(subject, body, image_path):
    global LAST_EMAIL_SENT
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg.attach(MIMEText(f"{body}\n\nTimestamp: {timestamp}", 'plain'))
        
        with open(image_path, 'rb') as f:
            msg.attach(MIMEImage(f.read(), name=image_path))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            print("⌛ Logging in to email account...")
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            print("⌛ Sending email...")
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        
        LAST_EMAIL_SENT = datetime.now()
        print("✅ Email sent successfully!")
    except smtplib.SMTPAuthenticationError:
        print("❌ SMTP Authentication Error: Check your email credentials or App Password.")
    except smtplib.SMTPConnectError:
        print("❌ SMTP Connect Error: Unable to connect to the SMTP server. Check your network or firewall settings.")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# ====== Detection Functions ======
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
        
        return ((mid_shoulder[0] + mid_hip[0])/2, 
                (mid_shoulder[1] + mid_hip[1])/2)
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

def is_glove_properly_worn(detection, landmarks, image_shape):
    """Enhanced glove detection logic with wrist landmarks"""
    try:
        left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image_shape[0])
        right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_shape[1],
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_shape[0])
        
        # Check proximity with increased threshold
        left_hand = is_near(detection, left_wrist, 80)
        right_hand = is_near(detection, right_wrist, 80)
        
        return left_hand or right_hand
    except:
        return False

def process_frame(image):
    global LAST_TERMINAL_OUTPUT, LAST_EMAIL_SENT 
    yolo_detections, yolo_class_ids, yolo_confidences = detect_with_yolo(image)
    detected_classes = set()
    
    # Process detections and draw boxes
    for i, detection in enumerate(yolo_detections):
        if yolo_confidences[i] >= 0.6:  # Lower confidence threshold
            class_id = int(yolo_class_ids[i])
            item_name = class_names[class_id]
            detected_classes.add(item_name)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, detection)
            color = (0, 255, 0)
            label = f"{class_names[class_id]} {yolo_confidences[i]:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Pose processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    output_image = image.copy()
    alerts = []
    current_time = datetime.now()

    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            output_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = pose_results.pose_landmarks.landmark
        
        # Keypoints
        head_kp = (landmarks[mp_pose.PoseLandmark.NOSE].x * image.shape[1],
                   landmarks[mp_pose.PoseLandmark.NOSE].y * image.shape[0])
        torso_center = get_torso_center(landmarks, image.shape)
        left_eye, right_eye = get_eye_centers(landmarks, image.shape)

        # Initialize PPE status
        required_ppe = ["Helmet", "Gloves", "Jacket", "Goggles"]
        ppe_status = {item: {"detected": False, "properly_worn": False} for item in required_ppe}

        # Update detection status
        for item in required_ppe:
            if item in detected_classes:
                ppe_status[item]["detected"] = True

        # Check proper wearing
        for i, detection in enumerate(yolo_detections):
            class_id = int(yolo_class_ids[i])
            item_name = class_names[class_id]
            confidence = yolo_confidences[i]
            
            if confidence < 0.6:  # Adjusted confidence threshold
                continue
                
            if item_name == "Helmet" and head_kp:
                ppe_status["Helmet"]["properly_worn"] = is_near(detection, head_kp, 100)
            elif item_name == "Gloves":
                ppe_status["Gloves"]["properly_worn"] = is_glove_properly_worn(detection, landmarks, image.shape)
            elif item_name == "Jacket" and torso_center:
                ppe_status["Jacket"]["properly_worn"] = is_near(detection, torso_center, 200)
            elif item_name == "Goggles" and left_eye and right_eye:
                ppe_status["Goggles"]["properly_worn"] = (
                    is_near(detection, left_eye, 50) or 
                    is_near(detection, right_eye, 50)
                )

        # Generate alerts
        for item in required_ppe:
            if not ppe_status[item]["detected"]:
                _add_alert(alerts, f"Missing {item}!", current_time)
            elif not ppe_status[item]["properly_worn"]:
                _add_alert(alerts, f"{item} not properly worn!", current_time)

        # Visualize wrist landmarks
        try:
            left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image.shape[0])
            right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image.shape[1],
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image.shape[0])
            cv2.circle(image, (int(left_wrist[0]), int(left_wrist[1])), 8, (255, 0, 0), -1)
            cv2.circle(image, (int(right_wrist[0]), int(right_wrist[1])), 8, (255, 0, 0), -1)
        except:
            pass

        # Terminal output
        if alerts and (datetime.now() - LAST_TERMINAL_OUTPUT) > TERMINAL_COOLDOWN:
            print("\n".join([f"❌ {alert}" for alert in alerts]))
            LAST_TERMINAL_OUTPUT = datetime.now()

        # Email handling
        if alerts and (datetime.now() - LAST_EMAIL_SENT) > EMAIL_COOLDOWN:
            output_path = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(output_path, image)
            print("⌛ Attempting to send email...")
            send_email_async("PPE Compliance Alert", "\n".join(alerts), output_path)
            LAST_EMAIL_SENT = datetime.now()

    return image

def detect_with_yolo(image):
    global LAST_SPEED_OUTPUT
    
    # Lower confidence threshold for better glove detection
    results = yolo_model.predict(
        source=image, 
        imgsz=640, 
        conf=0.4,  # Lowered from 0.5
        verbose=False
    )
    
    # Speed metrics
    if datetime.now() - LAST_SPEED_OUTPUT > TERMINAL_SPEED_COOLDOWN:
        if len(results) > 0 and hasattr(results[0], 'speed'):
            speed = results[0].speed
            print(f"Speed: {speed['preprocess']:.1f}ms preprocess, "
                  f"{speed['inference']:.1f}ms inference, "
                  f"{speed['postprocess']:.1f}ms postprocess")
        
        if len(results[0].boxes) > 0:
            detected_items = [f"{class_names[int(cls)]} {conf:.1f}%" 
                            for cls, conf in zip(results[0].boxes.cls.cpu().numpy(), 
                                               results[0].boxes.conf.cpu().numpy())]
            print(f"Detected: {', '.join(detected_items)}")
        
        LAST_SPEED_OUTPUT = datetime.now()
    
    return results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy(), results[0].boxes.conf.cpu().numpy()

def is_near(object_box, keypoint, threshold=100):
    """Check if the object is near the keypoint within a threshold."""
    x1, y1, x2, y2 = object_box
    kp_x, kp_y = keypoint
    return x1 - threshold < kp_x < x2 + threshold and y1 - threshold < kp_y < y2 + threshold

def main():
    print("Main")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {actual_width}x{actual_height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        output_frame = process_frame(frame)
        cv2.imshow('PPE Monitoring System - Press Q to exit', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()