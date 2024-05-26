# Importing the necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# Loading the dataset
dataset, info = tfds.load('food101', with_info=True, as_supervised=True)

# Preprocess the data by resizing and normalizing pixel value
def preprocess(image, label):
    image = tf.image.resize(image, (32, 32))
    image = image / 255.0
    return image, label

# .map() applies the function(preprocess) to each and every element
# (.batch(32)): Combines the processed elements into batches of 32. 
# This is useful for efficient training and evaluation.
# .prefetch() (.prefetch(1)): Prefetches 1 batch of data to be ready in the background
#  while the current batch is being processed by the model.
train_dataset = dataset['train'].map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = dataset['validation'].map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)



# adding autotune automatically gets the number of batches in data eg. prefetch(n)
# Without AUTOTUNE, 
# we would have to manually set the number of elements to prefetch or the number of parallel calls:



# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(101))  # since the number of data are 101

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Logits: The raw output values from the last layer of
# your neural network before any activation function is applied

     


# Train the model
model.fit(train_dataset, epochs=40, validation_data=test_dataset)

# applying early stopping
early_stop =tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

# Evaluate the model
model.evaluate(test_dataset, verbose=2)
