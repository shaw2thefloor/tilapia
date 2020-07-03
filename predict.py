
# Created by fshaw at 27/02/2019
import pandas as pd
import numpy as np
import os
import keras
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

model = keras.models.load_model("m.h5")

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20, zoom_range=0.15,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                   horizontal_flip=True, fill_mode="nearest")

proj_dir = Path('C:/Users/fshaw/Documents/PycharmProjects/tilapia/data/image_data/split/')

train_generator = train_datagen.flow_from_directory(proj_dir / "0" / "train",
                                                    target_size=(299, 299),
                                                    color_mode='rgb',
                                                    batch_size=10,
                                                    class_mode='categorical',
                                                    shuffle=True)

val_generator = train_datagen.flow_from_directory(proj_dir / "0" / "test",
                                                  target_size=(299, 299),
                                                  color_mode='rgb',
                                                  batch_size=10,
                                                  class_mode='categorical',
                                                  shuffle=True)


y_pred = model.predict_generator(train_generator)
y_pred_mx = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(train_generator.classes, y_pred_mx))
print('Classification Report')
target_names = os.listdir(proj_dir / "0" / 'test')
print(classification_report(train_generator.classes, y_pred_mx, target_names=target_names))
