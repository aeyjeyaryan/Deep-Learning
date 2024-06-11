import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
# The to_categorical function is used to convert class vectors (integers) to binary class matrices.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

dataset_path = r'C:\Users\sahil\OneDrive\Desktop\datasets\Food101'

img_size = (128, 128)

# Loading the first 10 images and their labels
images = []
labels = []
classes = os.listdir(os.path.join(dataset_path, 'images'))
for i, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, 'images', class_name)
    image_files = os.listdir(class_path)
    for j, image_file in enumerate(image_files[:1]):  # Only take one image per class
        if len(images) >= 10:
            break
        image_path = os.path.join(class_path, image_file)
        image = load_img(image_path, target_size=img_size)
        image = img_to_array(image)
        images.append(image)
        labels.append(i)
    if len(images) >= 10:
        break

# Converting into arrays
images = np.array(images)
labels = np.array(labels)

# Normalizing the images for a good fit
images = images / 255.0

# Converting to matrices
labels = to_categorical(labels, num_classes=len(classes))

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the mdel 
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluating the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Visualization of the images
predictions = model.predict(x_test)
for i in range(10):
    plt.imshow(x_test[i])
    plt.title(f'True: {np.argmax(y_test[i])}, Pred: {np.argmax(predictions[i])}')
    plt.show()
