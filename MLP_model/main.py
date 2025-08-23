import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import json

# Load label map
with open('label_mapping.json', 'r') as f:
    label_map = json.load(f)
label_to_index = label_map
index_to_label = {v: k for k, v in label_to_index.items()}

# Define model
class MLP(nn.Module):
    def __init__(self, input_size=42, num_classes=len(label_to_index)):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Normalize landmarks
def normalize_landmarks(landmark_list):
    base_x = landmark_list.landmark[0].x
    base_y = landmark_list.landmark[0].y
    norm = []
    for lm in landmark_list.landmark:
        norm.extend([lm.x - base_x, lm.y - base_y])
    return norm

# Load model
model = MLP()
model.load_state_dict(torch.load('mlp_model.pth'))
model.eval()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            norm_landmarks = normalize_landmarks(hand_landmarks)

            if len(norm_landmarks) == 42:
                input_tensor = torch.tensor([norm_landmarks], dtype=torch.float32)
                output = model(input_tensor)
                pred_index = torch.argmax(output, dim=1).item()
                pred_label = index_to_label[pred_index]

                cv2.putText(frame, f"{pred_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

    cv2.imshow('Hijaiyah Hand Sign Prediction', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
