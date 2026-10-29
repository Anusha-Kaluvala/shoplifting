import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from ultralytics import YOLO
import xgboost as xgb

def load_ground_truth_labels(csv_path):
    """Load the ground truth labels from the dataset."""
    return pd.read_csv(csv_path)

def evaluate_model(yolo_model_path, xgb_model_path, dataset_csv, images_dir):
    # Load YOLO model
    model_yolo = YOLO(yolo_model_path)

    # Load XGBoost model
    model_xgb = xgb.Booster()
    model_xgb.load_model(xgb_model_path)

    # Load ground truth
    ground_truth = load_ground_truth_labels(dataset_csv)

    y_true = []
    y_pred = []

    for _, row in ground_truth.iterrows():
        image_path = f"{images_dir}/{row['image_name']}"
        label = 1 if row['label'] == 'Normal' else 0

        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            continue

        frame = cv2.resize(frame, (1018, 600))

        # Perform YOLO inference
        results = model_yolo(frame, verbose=False)

        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for index, box in enumerate(bound_box):
                if conf[index] > 0.6:  # Confidence threshold
                    # Extract keypoints
                    data = {}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    df = pd.DataFrame(data, index=[0])
                    dmatrix = xgb.DMatrix(df)

                    # XGBoost prediction
                    sus = model_xgb.predict(dmatrix)
                    binary_prediction = (sus > 0.9).astype(int)  # 1: Normal, 0: Suspicious

                    y_true.append(label)
                    y_pred.append(binary_prediction[0])

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')

    print("Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return accuracy, precision, recall

# Example usage:
yolo_model_path = 'yolo11s-pose.pt'
xgb_model_path = 'trained_model.json'
dataset_csv = 'dataset.csv'
images_dir = 'images'  # Directory containing the images

evaluate_model(yolo_model_path, xgb_model_path, dataset_csv, images_dir)
