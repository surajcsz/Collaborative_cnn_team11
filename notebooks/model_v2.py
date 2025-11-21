################################################################
# Improved CNN model model_v2.py prepared by Suraj Kumar Singh #
################################################################

# IMPORT NECESSARY PACKAGES
import os
import json
import torch
import numpy as np
import pandas as pd
import opendatasets as od
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense

# STATIC DATA LABELS
LABELS=['Cat', 'Dog']
LABEL_MAP = {'cat':0, 'dog':1}

# IMPORT & TEST REQUIRED DATASET
od.download("https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition")
data = pd.read_csv("/content/dogs-vs-cats-redux-kernels-edition/sample_submission.csv")
print(data)
!unzip /content/dogs-vs-cats-redux-kernels-edition/train.zip
!unzip /content/dogs-vs-cats-redux-kernels-edition/test.zip

# PREPARE LABELLED TRAIN DATA
image_dir = '/content/train'
filenames = os.listdir(image_dir)
labels = [x.split('.')[0] for x in filenames]
data = pd.DataFrame({
    'filename': filenames, 
    'label': labels
    }
)

# SPLITTING DATASET AMONG TRAIN, VALIDATE & TEST DATASET
# 70% : TRAINING DATA ; 
# 15% : VALIDATION DATA &
# 15% : TESTING DATA
X_train, temp_data = train_test_split(
    data,
    test_size=0.3,
    stratify=data["label"],
    random_state=42
)

X_val, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    stratify=temp_data["label"],
    random_state=42
)

# IMAGE ARGUMENT STANDARDS
image_size = 128  
bat_size = 32     
channel = 3       

# TRAINING DATASET AUGUMENTATION
train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            channel_shift_range=0.2,
            fill_mode='nearest',
            horizontal_flip=True,
            rescale=1/255)

# PREPARING TRAINING & VALIDATION DATA FOR MODEL
train_generator = train_datagen.flow_from_dataframe(X_train,
                                                    directory = 'train/',
                                                    x_col= 'filename',
                                                    y_col= 'label',
                                                    batch_size = bat_size,
                                                    target_size = (image_size,image_size)
                                                   )
val_generator = test_datagen.flow_from_dataframe(X_val,
                                                 directory = 'train/',
                                                 x_col= 'filename',
                                                 y_col= 'label',
                                                 batch_size = bat_size,
                                                 target_size = (image_size,image_size),
                                                 shuffle=False
                                                )
                                                
# CREATING THE MODEL WITH LAYERS
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape = (image_size,image_size,channel)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(2,activation='softmax'))

model.summary()

# IMPLEMENTING LEARNING RATE REDUCTION & EARLY STOPPING
learning_rate_reduction = ReduceLROnPlateau(
    monitor = 'val_accuracy',
    patience = 2,
    factor = 0.5,
    min_lr = 0.0001,
    verbose = 1
)
early_stopping = EarlyStopping(
    monitor = 'val_loss', 
    patience = 3, 
    restore_best_weights = 'True', 
    verbose = 0
)

# CONFIGURING THE MODEL FOR EVALUATION
model.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy']
)

# TRAINING THE MODEL ON TRAINING & VALIDATION DATASET 
cat_dog = model.fit(
    train_generator, 
    validation_data = val_generator, 
    callbacks = [
        early_stopping, 
        learning_rate_reduction
    ], 
    epochs = 5
)

# PREPARE TEST DATASET
test_generator = test_datagen.flow_from_dataframe(
    X_test,
    directory = "test/",
    x_col = "filename",
    y_col = "label",
    target_size=(image_size, image_size),
    batch_size=bat_size,
    class_mode=None,
    shuffle=False
)

# PREDICTING THE RESULTS FROM THE MODEL
X_test['label'] = X_test['label'].map(LABEL_MAP)
y_test_true = X_test['label'].values

test_predict = model.predict(test_generator,verbose = 0)
y_test_pred = np.argmax(test_predict, axis=1)
        
# MODEL METRICS
report_dict = classification_report(y_test_true, y_test_pred, target_names=LABELS, output_dict=True)
json_report = json.dumps(report_dict, indent=4)

# SAVING THE MODEL & METRICS
with open('metrics_v2.json', 'w') as f:
        f.write(json_report)
model.save('model_v2.keras')

# MODEL COMPLETED 