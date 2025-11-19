# ============================================================
# 0. IMPORTS AND SETUP
# ============================================================

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Create folders based on required repo structure
ROOT = ".."   # because notebook is inside /notebooks

os.makedirs(f"{ROOT}/models", exist_ok=True)
os.makedirs(f"{ROOT}/results", exist_ok=True)
os.makedirs(f"{ROOT}/utils", exist_ok=True)

# ============================================================
# 1. DATASET LOADING  (LOCAL PATHS)
# ============================================================

DATA_DIR = f"{ROOT}/data"   # project_root/data

TRAIN_DIR = os.path.join(DATA_DIR, "training_set")
VAL_DIR   = os.path.join(DATA_DIR, "test_set")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# ============================================================
# 2. MODEL v1 — Transfer Learning (ResNet18)
# ============================================================

model = models.resnet18(pretrained=True)

# -------- UNFREEZE LAST TWO RESNET BLOCKS --------------------
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# -------- ADD DROPOUT + OUTPUT LAYER -------------------------
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, 1)
)

model = model.to(DEVICE)

# ============================================================
# LOSS, OPTIMIZER (with WEIGHT DECAY to reduce overfitting)
# ============================================================

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# ============================================================
# 3. TRAINING LOOP
# ============================================================

EPOCHS = 2
train_losses = []

print("\nTraining Model v1...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# ============================================================
# 4. EVALUATION (Accuracy, Precision, Recall, F1)
# ============================================================

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)

        outputs = torch.sigmoid(model(images)).cpu().numpy()
        preds = (outputs > 0.5).astype(int)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

print("\n========= METRICS (Model v1) =========")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

# ============================================================
# 5. SAVE MODEL WEIGHTS
# ============================================================

MODEL_PATH = f"{ROOT}/models/model_v1.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel v1 saved to: {MODEL_PATH}")

# ============================================================
# 6. SAVE METRICS TO JSON
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "train_loss_history": train_losses
}

METRIC_PATH = f"{ROOT}/results/metrics_v1.json"

with open(METRIC_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to: {METRIC_PATH}")

# ============================================================
# 7. DONE
# ============================================================

print("\nModel v1 training pipeline completed successfully!")
