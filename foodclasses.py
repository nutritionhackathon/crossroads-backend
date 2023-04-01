import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
from utils import add_regularization, get_data_generators, plot_model_results
from Models import TrainingCheckpoint, ModelTrain

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    rescale=1.0/255.0,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

TARGET_DIM = 300
BATCH_SIZE = 128

train_generator, validation_generator = get_data_generators(datagen, TARGET_DIM, BATCH_SIZE)

base_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(TARGET_DIM, TARGET_DIM, 3)
)

print('Layers in InceptionV3: ' + str(len(base_model.layers)))

preds = base_model.output
preds = tf.keras.layers.GlobalAveragePooling2D()(preds)
preds = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(preds)
preds = tf.keras.layers.BatchNormalization()(preds)
preds = tf.keras.layers.Dense(512, activation=tf.nn.relu)(preds)
preds = tf.keras.layers.BatchNormalization()(preds)
preds = tf.keras.layers.Dense(256, activation=tf.nn.relu)(preds)
preds = tf.keras.layers.BatchNormalization()(preds)
preds = tf.keras.layers.Dense(128, activation=tf.nn.relu)(preds)
preds = tf.keras.layers.Dense(11, activation=tf.nn.softmax)(preds)

model = tf.keras.models.Model(base_model.input, preds)

inceptionV3 = ModelTrain(model, TARGET_DIM, BATCH_SIZE)
inceptionV3.freeze_layers(9)

inceptionV3.model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['acc']
)

filepath="./models/inceptionV3-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

training_checkpoint = TrainingCheckpoint()

csv_logger = tf.keras.callbacks.CSVLogger(filename='./logs/inceptionV3_training.csv', append=True)

inceptionV3.model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, csv_logger, training_checkpoint],
    epochs=60
)

plot_model_results(9, model.history.history, 'acc')
plot_model_results(9, model.history.history, 'loss')

model = tf.keras.models.load_model('./models/inceptionV3-07-0.90.hdf5')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    rescale=1.0/255.0,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30
)

train_generator, validation_generator = get_data_generators(datagen, TARGET_DIM, BATCH_SIZE)

filepath="./models/inceptionV3-regularized2-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

training_checkpoint = TrainingCheckpoint()

csv_logger = tf.keras.callbacks.CSVLogger(filename='./logs/inceptionV3_regularize.csv', append=True)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', patience=2)

weight_regularization_layers = [model.layers[-2], model.layers[-6]]
model = add_regularization(model, weight_regularization_layers, regularizer=tf.keras.regularizers.l2(l=0.003))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['acc']
)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, csv_logger, training_checkpoint, early_stopping],
    epochs=20
)

model = tf.keras.models.load_model('./models/inceptionV3-regularized-04-0.77.hdf5')

preds = model.layers[-8].output
preds = tf.keras.layers.Dropout(rate=0.4)(preds)
preds = model.layers[-7](preds)
preds = model.layers[-6](preds)
preds = model.layers[-5](preds)
preds = model.layers[-4](preds)
preds = tf.keras.layers.Dropout(rate=0.3)(preds)
preds = model.layers[-3](preds)
preds = model.layers[-2](preds)
preds = model.layers[-1](preds)

model = tf.keras.models.Model(model.input, preds)

#print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['acc']
)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, csv_logger, training_checkpoint],
    epochs=15
)

model = tf.keras.models.load_model('./models/inceptionV3-07-0.90.hdf5')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    rescale=1.0/255.0
)


def evaluate_model(model, target_dim, batch_size):
    test_generator = datagen.flow_from_directory(
        directory='./evaluation/',
        target_size=(target_dim, target_dim),
        batch_size=batch_size,
        shuffle=False
    )

    return model.evaluate_generator(test_generator, steps=test_generator.samples // batch_size)


evaluation = evaluate_model(model, 300, 128)
print(evaluation)

classToName = {0:'Bread', 1:'Dairy product', 2: 'Dessert', 3: 'Egg', 4: 'Fried food', 5:'Meat', 6: 'Noodles-Pasta', 7: 'Rice', 8:'Seafood', 9:'Soup', 10: 'Vegetable-Fruit'}

image = tf.keras.preprocessing.image.load_img('bread_milk.webp', target_size=(300, 300))
image = tf.keras.preprocessing.image.img_to_array(image)

image = image.reshape(1, 300, 300, 3)
image = tf.keras.applications.vgg19.preprocess_input(image)
image = image / 255

print(model.predict(image))
print(classToName[np.argmax(model.predict(image))])

