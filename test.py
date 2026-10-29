import os
import cv2
import uuid
import asyncio
import numpy as np
import base64
import json
import eventlet
eventlet.monkey_patch() 
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import xgboost as xgb
import pandas as pd
import threading
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize models and configurations
model_yolo = YOLO('src/yolo11s-pose.pt')
xgb_model = xgb.Booster()
xgb_model.load_model('src/trained_model.json')

# Ensure directories exist
UPLOAD_FOLDER = 'uploads'
SUSPICIOUS_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUSPICIOUS_FOLDER, exist_ok=True)

def process_video_stream(filename, sid):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)
    
    # Prepare video details for frontend
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Send initial video metadata
    socketio.emit('video_metadata', {
        'total_frames': total_frames,
        'fps': fps
    }, room=sid)
    
    frame_count = 0
    suspicious_timestamps = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        # Process every 3rd frame to reduce computational load
        if frame_count % 3 != 0:
            continue
        
        # Resize the frame
        frame = cv2.resize(frame, (1018, 600))
        
        # Run YOLOv8 on the frame
        results = model_yolo(frame, verbose=False)
        
        suspicious_in_frame = False
        
        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()
            
            for index, box in enumerate(bound_box):
                if conf[index] > 0.55:
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Prepare data for XGBoost prediction
                    data = {}
                    for j in range(len(keypoints[index])): 
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]
                    
                    df = pd.DataFrame(data, index=[0])
                    dmatrix = xgb.DMatrix(df)
                    
                    # Make prediction
                    sus = xgb_model.predict(dmatrix)
                    binary_predictions = (sus > 0.5).astype(int)
                    
                    if binary_predictions == 0:  # Suspicious
                        suspicious_in_frame = True
                        
                        # Extract and save suspicious region
                        suspicious_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                        susp_filename = os.path.join(SUSPICIOUS_FOLDER, f'susp_{uuid.uuid4()}.jpg')
                        cv2.imwrite(susp_filename, suspicious_frame)
                        
                        # Calculate timestamp
                        timestamp = frame_count / fps
                        suspicious_timestamps.append({
                            'time': timestamp,
                            'image': os.path.basename(susp_filename)
                        })
        
        # Emit frame progress and any suspicious detections
        socketio.emit('processing_update', {
            'current_frame': frame_count,
            'total_frames': total_frames,
            'suspicious_found': suspicious_in_frame
        }, room=sid)
        
        # If suspicious activity, send details
        if suspicious_in_frame:
            socketio.emit('suspicious_detection', {
                'timestamps': suspicious_timestamps
            }, room=sid)
    
    # Finalize processing
    socketio.emit('processing_complete', {
        'total_suspicious': len(suspicious_timestamps),
        'timestamps': suspicious_timestamps
    }, room=sid)
    
    cap.release()

@app.route('/')
def index():
    return render_template('index1.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
video_buffers=[]
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Emit a WebSocket event to notify frontend about the successful upload
    socketio.emit('file_uploaded', {'filename': filename, 'status': 'success'})
    return jsonify({"message": "File uploaded successfully"}), 200

    socketio.start_background_task(process_video_stream, filename, request.sid)
    # Respond to client that upload is complete
    emit('upload_complete', {'filename': filename})

if __name__ == '__main__':
    socketio.run(app, debug=True)