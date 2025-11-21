########################################################################
# Validation of model_v1.pth from USER-1 prepared by Suraj Kumar Singh #
########################################################################

#IMPORTING NECESSARY PACKAGES
import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import opendatasets as od
from torchvision import models
from torch.utils.data import DataLoader
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense

# IMPORTING THE DATASET
od.download("https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition")
data = pd.read_csv("/content/dogs-vs-cats-redux-kernels-edition/sample_submission.csv")
print(data)
!unzip /content/dogs-vs-cats-redux-kernels-edition/train.zip
!unzip /content/dogs-vs-cats-redux-kernels-edition/test.zip

# STATIC LABEL VALUES
LABELS = ['Cat', 'Dog']

# PREPARING MODEL CLASS TO LOAD
model = models.resnet18(pretrained=True)
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, 1)
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)

# LOAD AND EVALUATING THE MODEL
model.load_state_dict(torch.load("/content/model_v1.pth",DEVICE))
model.eval()

# PREPARING THE TEST DATASET
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder("/content/train", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# PREDICT FROM THE MODEL
correct = 0
total = 0
y_true, y_pred = [], []
with torch.no_grad(): 
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# METRICS REPORT 
report_dict=classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
json_report = json.dumps(report_dict, indent=4)

# REPORT TO FILE
with open('test_v1_user2.json', 'w') as f:
    f.write(json_report)
print(json_report)