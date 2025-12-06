'''
Training a CNN model on existing images for RL as above but tweaking model.

Less epochs - 10
Smaller convolutional layers

'''

import os
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, MaxPooling2D, Conv2D, Concatenate, Embedding, Reshape, Flatten, Activation, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


# HEIGHT = 240
# WIDTH = 320
HEIGHT = 120
WIDTH = 160

MAX_STEER_DEGREES = 40

data_dir = '/home/ana/Documents/Architecture_Transformers/dataset_carla'

batch_size = 64   # 16
image_shape = (HEIGHT, WIDTH, 3)


# Create a custom data generator
def custom_data_generator(image_files, batch_size):
    num_samples = len(image_files)
    while True:
        indices = np.random.randint(0, num_samples, batch_size)
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            image_path = image_files[idx]
            raw_label = os.path.basename(image_path).split('.png')[0].split('_')[2]
            raw_label = raw_label.replace('p', '.')   # +0p034 → +0.034
            label = float(raw_label)

            if label > MAX_STEER_DEGREES:
                label = MAX_STEER_DEGREES
            elif label < -MAX_STEER_DEGREES:
                label = -MAX_STEER_DEGREES
            label = float(label)/MAX_STEER_DEGREES

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (WIDTH, HEIGHT))   # Aqui pode se comentar
            image = image / 255.0

            batch_images.append(image)
            batch_labels.append(label)

        yield np.array(batch_images, dtype=np.float32), np.array(batch_labels, dtype=np.float32)



def create_model():
    image_input = Input(shape=image_shape)

    x = Conv2D(64, (6,6), activation='relu', padding='same')(image_input) # shape=(240,320,64)
    x = MaxPooling2D((2,2))(x)  # shape=(120,160,64)
    
    x = Conv2D(64, (6,6), activation='relu', padding='same')(x)  #shape=(120,160,64)
    x = MaxPooling2D((2,2))(x)  #shape=(60,80,64)
    
    x = Conv2D(64, (6,6), activation='relu', padding='same')(x)  #shape=(60,80,64)
    x = MaxPooling2D((2,2))(x)  # shape=(30,40,64)
    
    x = Conv2D(64, (6,6), activation='relu', padding='same')(x)  # shape=(30,40,64)
    x = MaxPooling2D((2,2))(x)  # shape=(15,20,64)

    x = Dense(8, activation='relu', activity_regularizer=regularizers.L2(1e-5))(x)  # shape=(15,20,8)
    x = Dropout(0.2)(x)  # shape=(15,20,8)
    x = Dense(4, activation='relu', activity_regularizer=regularizers.L2(1e-5))(x)  # shape=(15,20,4)
    x = Dropout(0.2)(x)  # shape=(15,20,4)

    x = Flatten(name='embedding_layer')(x) # shape=(15*20*4,)   

    output = Dense(1)(x)  # shape=(1,)
    return Model(inputs=image_input, outputs=output)



image_files = []    # Get a list of image file paths and labels
for file in os.listdir(data_dir):
    if file.endswith('.png'):
        image_files.append(os.path.join(data_dir, file)) 

random.shuffle(image_files)

# Split the data into training and validation sets
split_index = int(len(image_files) * 0.8)  # 80% for training, 20% for validation
train_files = image_files[:split_index]
val_files = image_files[split_index:]


# Create data generators for training and validation
train_generator = custom_data_generator(train_files, batch_size)
val_generator = custom_data_generator(val_files, batch_size)

model = create_model()
model.summary()
model.compile(loss='MSE', optimizer='adam')

# Train the model
model.fit(train_generator, steps_per_epoch=len(train_files) // batch_size, epochs=10,
          validation_data=val_generator, validation_steps=len(val_files) // batch_size)

#This is what actually saves the model to be used in RL training
# desired_layer_output  = model.get_layer('embedding_layer').output
desired_layer_output  = model.get_layer('embedding_layer').output  
model_to_save = Model(inputs=model.input, outputs=desired_layer_output)

# Save the new model - make sure you use your RL env to use this name if you use
model_to_save.save('CNN_image_model.h5')
model_to_save.summary()