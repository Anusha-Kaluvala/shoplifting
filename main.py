from flask import Flask, Response, render_template, request
from flask_socketio import SocketIO
import cv2
import os
import base64
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import cvzone
import pygame
import time  # Add this import at the top

# Initialize Pygame
pygame.init()
pygame.mixer.init()

app = Flask(__name__)
socket = SocketIO(app)
UPLOAD_FOLDER = "uploads"
BEEP_SOUND_FILE = "beep.wav"

# Create a fallback beep function if sound file fails
def play_beep():
    try:
        # Attempt to load the beep.wav sound file
        beep_sound = pygame.mixer.Sound(BEEP_SOUND_FILE)
        beep_sound.play()
    except pygame.error:
        try:
            import winsound  # Ensure this is only imported on Windows
            winsound.Beep(1000, 500)  # 1000 Hz, 500 ms
        except ImportError:
            print("No sound method available")

def detect_shoplifting_stream(video_path):
    # Load YOLOv8 model (replace with the actual path to your YOLOv8 model)
    model_yolo = YOLO('yolo11s-pose.pt')
    # Load the trained XGBoost model (replace with the actual path to your XGBoost model)
    model = xgb.Booster()
    model.load_model('trained_model.json')

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    i = 0
    count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        count += 1
        if count % 5 != 0:
            continue

        frame = cv2.resize(frame, (1018, 600))

        results = model_yolo(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        alert_triggered = False

        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for index, box in enumerate(bound_box):
                if conf[index] > 0.6:
                    x1, y1, x2, y2 = box.tolist()

                    data = {}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    df = pd.DataFrame(data, index=[0])
                    dmatrix = xgb.DMatrix(df)
                    sus = model.predict(dmatrix)
                    binary_predictions = (sus > 0.9).astype(int)

                    if binary_predictions == 0:  # Suspicious
                        alert_triggered = True
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.rectangle(frame,(int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cvzone.putTextRect(frame,"Suspicious",(int(x1), int(y1) - 10), 1, 1)
                        cvzone.putTextRect(annotated_frame, "Suspicious", (int(x1), int(y1) - 10), 1, 1)
                    else:
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cvzone.putTextRect(annotated_frame, "Normal", (int(x1), int(y1) - 10), 1, 1)

        if alert_triggered:
            i += 1
            if i%5==0:
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (1018, 600), (0, 0, 0), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
                cvzone.putTextRect(
                    annotated_frame,
                    "ALERT: Suspicious Activity Detected!",
                    (100, 300),
                    scale=2,
                    thickness=2,
                    colorR=(0, 0, 255),
                    offset=20
                )
                play_beep()
                # Convert annotated_frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                frame_base64 = base64.b64encode(frame).decode('utf-8')
                timestamp = int(time.time())  # Get current timestamp
                socket.emit('frame', {'image': frame_base64, 'id': timestamp})  # Emit the frame with a unique ID

        # Encode the frame as JPEG and yield
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files.get('video')
    if not video:
        return "No video uploaded.", 400

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    global uploaded_video_path
    uploaded_video_path = video_path  # Save path for streaming
    return "Video uploaded successfully.", 200

@app.route('/video_feed')
def video_feed():
    # Use the uploaded video path instead of a hardcoded path
    video_path = uploaded_video_path  # Use the global variable
    return Response(detect_shoplifting_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Start the Flask app with SocketIO
    socket.run(app, debug=True)
