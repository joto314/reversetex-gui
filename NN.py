import pickle
import tensorflow as tf
from tensorflow import keras

num_labels = 369

data = open('traina.pickle', 'rb')
xtrain, ytrain, trainsamples = pickle.load(data)

#1 4 conv layer
model1 = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same', input_shape=(32,32,1)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(32, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),

    keras.layers.Conv2D(64, 7, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, 7, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),

    keras.layers.Flatten(),

    keras.layers.Dense(2048),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(2048),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(num_labels),
    keras.layers.Activation('softmax'),
])

model1.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

#2 6 conv layer
model2 = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same', input_shape=(32,32,1)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(32, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),

    keras.layers.Conv2D(32, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(32, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),

    keras.layers.Conv2D(64, 7, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, 7, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),

    keras.layers.Flatten(),

    keras.layers.Dense(2048),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(2048),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(num_labels),
    keras.layers.Activation('softmax'),
])

model2.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

#3 8 conv layer 
model3 = keras.Sequential([
    keras.layers.Conv2D(32, 2, padding='same', input_shape=(32,32,1)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(32, 2, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),

    keras.layers.Conv2D(32, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(32, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),

    keras.layers.Conv2D(64, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),

    keras.layers.Conv2D(128, 5, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(128, 5, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),

    keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='same'),

    keras.layers.Flatten(),

    keras.layers.Dense(2048),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(2048),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(num_labels),
    keras.layers.Activation('softmax'),
])

model3.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy', 'top_k_categorical_accuracy'])


history = model1.fit(xtrain.reshape(trainsamples,32,32,1), ytrain, workers=4, epochs=100, batch_size=100, validation_split=0.05)
model1.save('model1') 


history = model2.fit(xtrain.reshape(trainsamples,32,32,1), ytrain, workers=4, epochs=100, batch_size=100, validation_split=0.05)
model2.save('model2') 


history = model3.fit(xtrain.reshape(trainsamples,32,32,1), ytrain, workers=4, epochs=100, batch_size=100, validation_split=0.05)
model3.save('model3') 