import os
import random
import json
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

import mediapipe as mp

# ---------------------------
# Configuration
# ---------------------------
dataset_path = './arabic_sign_language/RGB ArSL dataset'
batch_size = 32
epochs = 100
input_size = 42
learning_rate = 0.0001
model_save_path = 'hijaiyah_mlp_best.pth'
label_map_path = 'label_mapping.json'
patience = 5

# ---------------------------
# Helper Functions
# ---------------------------
def normalize_landmarks(landmark_list):
    base_x = landmark_list.landmark[0].x
    base_y = landmark_list.landmark[0].y
    norm = []
    for lm in landmark_list.landmark:
        norm.extend([lm.x - base_x, lm.y - base_y])
    return norm

def augment_landmarks(landmarks):
    if random.random() < 0.5:
        for i in range(0, len(landmarks), 2):
            landmarks[i] = -landmarks[i]

    angle = random.uniform(-15, 15) * np.pi / 180
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    for i in range(0, len(landmarks), 2):
        x, y = landmarks[i], landmarks[i+1]
        landmarks[i] = cos_val * x - sin_val * y
        landmarks[i+1] = sin_val * x + cos_val * y
    return landmarks

# ---------------------------
# Data Preparation
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

X_data, y_labels = [], []
label_to_index = {}
current_index = 0

for label_name in os.listdir(dataset_path):
    label_folder = os.path.join(dataset_path, label_name)
    if not os.path.isdir(label_folder):
        continue

    if label_name not in label_to_index:
        label_to_index[label_name] = current_index
        current_index += 1

    for img_file in tqdm(os.listdir(label_folder), desc=f"Processing {label_name}"):
        img_path = os.path.join(label_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            norm_landmarks = normalize_landmarks(landmarks)
            if len(norm_landmarks) == input_size:
                X_data.append(norm_landmarks)
                y_labels.append(label_to_index[label_name])
                X_data.append(augment_landmarks(norm_landmarks.copy()))
                y_labels.append(label_to_index[label_name])

# ---------------------------
# Dataset
# ---------------------------
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

train_dl = DataLoader(LandmarkDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_dl = DataLoader(LandmarkDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
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

model = MLP(input_size=input_size, num_classes=len(label_to_index))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ---------------------------
# Training Loop
# ---------------------------
best_acc = 0
trigger_times = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_dl:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y_batch.tolist())
            y_pred.extend(preds.tolist())

    acc = accuracy_score(y_true, y_pred)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {acc*100:.2f}%")

    if acc > best_acc + 0.001:
        best_acc = acc
        trigger_times = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# ---------------------------
# Save Label Map
# ---------------------------
with open(label_map_path, 'w') as f:
    json.dump(label_to_index, f)

print(f"\nModel saved to {model_save_path}")
print(f"Label mapping saved to {label_map_path}")
