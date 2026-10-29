import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone
import pygame

# Initialize pygame mixer for sound
pygame.mixer.init()

# Path to the beep sound file (ensure this file exists)
BEEP_SOUND_FILE = "anu/beep.wav"

# Create a fallback beep function if sound file fails
def play_beep():
    try:
        # Attempt to load the beep.wav sound file
        beep_sound = pygame.mixer.Sound(BEEP_SOUND_FILE)
        beep_sound.play()
    except pygame.error:
        # If loading fails, print an error message
        print(f"Error loading {BEEP_SOUND_FILE}. Please ensure the file is present and valid.")
        # Optionally, you could try using a default Python method to generate a beep
        # For example: print("\a") or use winsound on Windows
        try:
            import winsound
            winsound.Beep(1000, 500)  # 1000 Hz, 500 ms
        except ImportError:
            print("No sound method available")

# Define the path to the video file
video_path = "anu/susp1.mp4"

def detect_shoplifting(video_path):
    # Load YOLOv8 model (replace with the actual path to your YOLOv8 model)
    model_yolo = YOLO('anu/yolo11s-pose.pt')

    # Load the trained XGBoost model (replace with the actual path to your XGBoost model)
    model = xgb.Booster()
    model.load_model('anu/trained_model.json')

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print(f"Total Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Warning: Frame could not be read. Skipping.")
            break  # Stop the loop if no frame is read

        count += 1
        if count % 3 != 0:
            continue

        # Resize the frame
        frame = cv2.resize(frame, (1018, 600))

        # Run YOLOv8 on the frame
        results = model_yolo(frame, verbose=False)

        # Visualize the YOLO results on the frame
        annotated_frame = results[0].plot(boxes=False)

        alert_triggered = False  # Track if an alert is triggered in this frame

        for r in results:
            bound_box = r.boxes.xyxy  # Bounding box coordinates
            conf = r.boxes.conf.tolist()  # Confidence levels
            keypoints = r.keypoints.xyn.tolist()  # Keypoints for human pose

            for index, box in enumerate(bound_box):
                if conf[index] > 0.55:  # Threshold for confidence score
                    x1, y1, x2, y2 = box.tolist()

                    # Prepare data for XGBoost prediction
                    data = {}
                    for j in range(len(keypoints[index])): 
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    # Convert the data to a DataFrame
                    df = pd.DataFrame(data, index=[0])

                    # Prepare data for XGBoost prediction
                    dmatrix = xgb.DMatrix(df)

                    # Make prediction using the XGBoost model
                    sus = model.predict(dmatrix)
                    binary_predictions = (sus > 0.5).astype(int)
                    print(f'Prediction: {binary_predictions}')

                    # Annotate the frame based on prediction (0 = Suspicious, 1 = Normal)
                    if binary_predictions == 0:  # Suspicious
                        alert_triggered = True  # Trigger the alert
                        # Draw red rectangle and display "Suspicious"
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cvzone.putTextRect(annotated_frame, "Suspicious", (int(x1), int(y1) - 10), 1, 1)

                    else:  # Normal
                        # Draw green rectangle and display "Normal"
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cvzone.putTextRect(annotated_frame, "Normal", (int(x1), int(y1) - 10), 1, 1)

        # Add a popup alert overlay if a suspicious activity is detected
        if alert_triggered:
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (1018, 600), (0, 0, 0), -1)  # Black semi-transparent overlay
            alpha = 0.6  # Transparency factor
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

            # Display alert text
            cvzone.putTextRect(
                annotated_frame,
                "ALERT: Suspicious Activity Detected!",
                (100, 300),  # Position
                scale=2,
                thickness=2,
                colorR=(0, 0, 255),  # Red color
                offset=20
            )

            # Play beep sound to alert
            play_beep()

        # Show the annotated frame in a window
        cv2.imshow('Frame', annotated_frame)

        # Press 'q' to stop the video early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()

    # Close all OpenCV windows after processing is complete
    cv2.destroyAllWindows()

# Call the function with the video path
detect_shoplifting(video_path)
