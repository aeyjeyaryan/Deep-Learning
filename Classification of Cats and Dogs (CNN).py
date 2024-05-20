import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import UnidentifiedImageError

# Constants
IMAGE_SIZE = (64, 64)  
BATCH_SIZE = 32
DATASET_DIR = r'C:\Users\sahil\OneDrive\Desktop\datasets\PetImages'  

def load_and_preprocess_image(image_path):
    try:
        image = load_img(image_path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        image = image.astype(np.float32) / 255.0  # I used float32 because of less memory
        return image
    except UnidentifiedImageError:
        print(f"Cannot identify image file {image_path}. Skipping.")
        return None

# Had to look up on stack overflow for this block
def load_dataset(dataset_dir):
    images = []
    labels = []
    for label, subdir in enumerate(['Cat', 'Dog']):
        subdir_path = os.path.join(dataset_dir, subdir)
        if not os.path.exists(subdir_path):
            raise FileNotFoundError(f"Directory {subdir_path} not found. Please check the path and try again.")
        for filename in os.listdir(subdir_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(subdir_path, filename)
                image = load_and_preprocess_image(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset
images, labels = load_dataset(DATASET_DIR)

indices = np.arange(len(images))
np.random.shuffle(indices)
images, labels = images[indices], labels[indices]

split = int(0.8 * len(images))
train_images, train_labels = images[:split], labels[:split]
val_images, val_labels = images[split:], labels[split:]

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE)
val_gen = ImageDataGenerator().flow(val_images, val_labels, batch_size=BATCH_SIZE)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
EPOCHS = 20   #I should have set it to 50

history = model.fit(
    train_gen,
    steps_per_epoch=len(train_images) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=len(val_images) // BATCH_SIZE
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_gen)
print(f'Validation accuracy: {val_accuracy:.2f}')

# Save the model
model.save('cats_vs_dogs_cnn_model.h5')

# Function to preprocess image from user input
def preprocess_image_from_input(image_path):
    image = load_and_preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  
    return image

# Load the trained model
model = load_model('cats_vs_dogs_cnn_model.h5')

# Ask user for image input and make predictions
image_path = input("Enter the path of the image to classify: ")
preprocessed_image = preprocess_image_from_input(image_path)

# Predict the class (0 for cat, 1 for dog)
prediction = model.predict(preprocessed_image)
class_label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

print(f'The image is of a: {class_label}')
