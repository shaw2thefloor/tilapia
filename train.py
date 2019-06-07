# Created by fshaw at 27/02/2019
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers

# base_model = MobileNet(weights='imagenet',include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
# base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_tensor=None,
#                                                  input_shape=None, pooling='max')
base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
                                            pooling='max')
num_classes = 30
'''
x = base_model.output
#x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(
    x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(512, activation='relu')(x)  # dense layer 2
preds = Dense(num_classes, activation='softmax')(x)  # final layer with softmax activation
model = Model(inputs=x, outputs=preds)
# specify the inputs
# specify the outputs
# now a model has been created based on our architecture
'''
# Top Model Block
x = base_model.output
x = Dense(512, activation='relu')(
    x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(512, activation='relu')(x)  # dense layer 2
x = Dense(512, activation='relu')(x)  # dense layer 2
# x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)

# add your top layer block to your base model
model = Model(base_model.input, predictions)
print("num layers: " + str(len(model.layers)))

for layer in model.layers[:19]:
    layer.trainable = False
for layer in model.layers[19:]:
    layer.trainable = True
print(model.summary())
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20, zoom_range=0.15,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                   horizontal_flip=True, fill_mode="nearest")  # included in our dependencies
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('/home/fshaw/Documents/fish/images/split/train',
                                                    target_size=(299, 299),
                                                    color_mode='rgb',
                                                    batch_size=10,
                                                    class_mode='categorical',
                                                    shuffle=True)
val_generator = train_datagen.flow_from_directory('/home/fshaw/Documents/fish/images/split/test',
                                                  target_size=(299, 299),
                                                  color_mode='rgb',
                                                  batch_size=10,
                                                  class_mode='categorical',
                                                  shuffle=True)

op = optimizers.SGD(lr=0.1, momentum=0.1, decay=0.01, nesterov=False)
adam = optimizers.Adam(lr=0.00001)
model.compile(optimizer=adam, loss='poisson', metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
print("Beginning Training")
step_size_train = train_generator.n // train_generator.batch_size
validation_steps = val_generator.n // val_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=1000,
                    validation_data=val_generator,
                    validation_steps=validation_steps,
                    )
print("Saving Model")
model.save("./trained_model.h5", overwrite=True, include_optimizer=True)