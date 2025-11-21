#Improved CNN model model_v2.py prepared by Suraj Kumar Singh

#Importing all the necessary packages
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

#Static Label Data
labels=['Cat', 'Dog']
label_mapping = {'cat':0, 'dog':1}

#Downloading & Testing the Dataset
od.download("https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition")
data = pd.read_csv("/content/dogs-vs-cats-redux-kernels-edition/sample_submission.csv")
print(data)

#Unzipping the Dataset
!unzip /content/dogs-vs-cats-redux-kernels-edition/train.zip
!unzip /content/dogs-vs-cats-redux-kernels-edition/test.zip

#Preparing the Labelled Train data
image_dir = '/content/train'
filenames = os.listdir(image_dir)
labels = [x.split('.')[0] for x in filenames]
data = pd.DataFrame({
    'filename': filenames, 
    'label': labels
    }
)

#Creating the Training, Validating and Testing Dataset
#70% train data; 15% validation data and 15% test data
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

#Standardizing the Images
image_size = 128  
bat_size = 32     
channel = 3       

#Augumenting the Training Data Generator
train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            channel_shift_range=0.2,
            fill_mode='nearest',
            horizontal_flip=True,
            rescale=1/255)

#Preparing Training and Validation Image Data for the Model
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
                                                
#Creating the Model and checking the Layers
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

#Preparing Reducing Learning Rate and Early Stopping
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

#Configuring the model for training and evaluation
model.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy']
)

#Training the model
cat_dog = model.fit(
    train_generator, 
    validation_data = val_generator, 
    callbacks = [
        early_stopping, 
        learning_rate_reduction
    ], 
    epochs = 5
)

#Preparing the Testing Data Set
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

#Predicting the Test results from the model
X_test['label'] = X_test['label'].map(label_mapping)
y_test_true = X_test['label'].values

test_predict = model.predict(test_generator,verbose = 0)
y_test_pred = np.argmax(test_predict, axis=1)
        
#Checking the metrics for the model
report_dict = classification_report(y_test_true, y_test_pred, target_names=labels, output_dict=True)
json_report = json.dumps(report_dict, indent=4)

with open('metrics_v2.json', 'w') as f:
        f.write(json_report)

#Save the Model weights
model.save('model_v2.keras')

#Successfully completed and  saved the CNN Model