# Created by fshaw at 27/02/2019
import os
import keras
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import Precision, Recall, TrueNegatives, TruePositives, FalseNegatives, FalsePositives
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
                                            pooling='max')
num_classes = 11
num_folds = 10
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
x = Dense(256, activation='relu')(x)
# x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
# x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
# x = GlobalAveragePooling2D()(x)

# x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)

# add your top layer block to your base model
model = Model(base_model.input, predictions)
print("num layers: " + str(len(model.layers)))

for layer in model.layers[:21]:
    layer.trainable = False
for layer in model.layers[22:]:
    layer.trainable = True
print(model.summary())
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20, zoom_range=0.15,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                   horizontal_flip=True, fill_mode="nearest")
# included in our dependencies
# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies


proj_dir = Path('C:/Users/fshaw/Documents/PycharmProjects/tilapia/data/image_data/split/')
accuracies = list()
accuracies_file_path = Path('C:/Users/fshaw/Documents/PycharmProjects/tilapia/accuracies.txt')
try:
    os.remove(accuracies_file_path)
except:
    pass

# for dir in os.listdir(proj_dir):

# if "all" in dir:
#    print(dir)
#    break

train_generator = train_datagen.flow_from_directory(proj_dir / "0" / "train",
                                                    target_size=(299, 299),
                                                    color_mode='rgb',
                                                    batch_size=40,
                                                    class_mode='categorical',
                                                    shuffle=True)
val_generator = train_datagen.flow_from_directory(proj_dir / "0" / "test",
                                                  target_size=(299, 299),
                                                  color_mode='rgb',
                                                  batch_size=40,
                                                  class_mode='categorical',
                                                  shuffle=True)

op = optimizers.SGD(lr=0.1, momentum=0.1, decay=0.01, nesterov=False)
adam = optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=[Precision(), Recall(), TruePositives(), FalsePositives(), FalseNegatives(), TrueNegatives()])
# Adam optimizer
# loss function will be categorical cross entropy

print("Beginning Training")
step_size_train = train_generator.n // train_generator.batch_size
validation_steps = val_generator.n // val_generator.batch_size
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=step_size_train,
                              epochs=100,
                              validation_data=val_generator,
                              validation_steps=validation_steps,

                              )

print(history.history)

'''
# y_pred always seems to be one class
y_pred = model.predict_generator(val_generator)
y_pred_mx = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(val_generator.classes, y_pred_mx))
print('Classification Report')
# target_names = os.listdir(proj_dir / 'test')
print(classification_report(val_generator.classes, y_pred_mx))
'''
