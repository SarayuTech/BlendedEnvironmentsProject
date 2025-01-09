import os
import cv2
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
from keras import utils, models, layers, regularizers
from keras import callbacks  # EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Directory for the Separated_Frames dataset
separated_frames_directory = 'Separated_Frames'
img_size = (32, 32)  # Reduced image size to save memory

# Get class labels from the directories in Separated_Frames
class_labels = os.listdir(separated_frames_directory)
num_classes = len(class_labels)

images = []
labels = []
image_paths = []
count = 0

# Iterate through each class and its respective images
for label in class_labels:
    class_dir = os.path.join(separated_frames_directory, label)
    for file_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, file_name)
        
        # Read and resize image
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)  # Resize image to smaller size
            images.append(img)
            labels.append(label)
            image_paths.append(img_path)
        
        count += 1
        print(f"Processed {count} images")

images = np.array(images)
labels = np.array(labels)

# Normalize images
images = images / 255.0  # Normalize to range [0, 1]

# Encoding labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = utils.to_categorical(labels, num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val, train_paths, val_paths = train_test_split(images, labels, image_paths, test_size=0.2, random_state=15)

print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of validation samples: {X_val.shape[0]}")

# Creating directories for validation data
validation_dir = 'validation'
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

for label in class_labels:
    label_dir = os.path.join(validation_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Copy validation images to validation directory
for img_path in val_paths:
    one_hot_encoded_label = labels[val_paths.index(img_path)]
    encoded_label = np.argmax(one_hot_encoded_label)  # Get the index of the max value in one-hot encoding
    label_name = label_encoder.inverse_transform([encoded_label])[0]  # Ensure it is a single value
    label_dir = os.path.join(validation_dir, label_name)
    dest_path = os.path.join(label_dir, os.path.basename(img_path))
    shutil.copy(img_path, dest_path)

# Custom function for data augmentation
def augment_image(image, label):
    # Apply random transformations
    image = tf.image.random_flip_left_right(image)  # Random horizontal flip
    image = tf.image.random_flip_up_down(image)     # Random vertical flip
    image = tf.image.random_brightness(image, max_delta=0.3)  # Random brightness
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Random contrast
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Random saturation
    image = tf.image.random_hue(image, max_delta=0.1)  # Random hue
    image = tf.image.resize(image, [img_size[0], img_size[1]])  # Ensure the image is resized back to target size

    return image, label

# Creating TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# Apply augmentation to the training dataset
train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch the training dataset
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Batch the validation dataset (no augmentation)
val_dataset = val_dataset.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Set up early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,  # Stop after 3 epochs of no improvement
    restore_best_weights=True  # Restore the weights from the best epoch
)

# Model definition with L2 regularization
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_size[0], img_size[1], 3),
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.1),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with early stopping
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

# Evaluate model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save model
model.save('emotion_model.h5')

# Plotting the training and validation accuracy graph and save as jpeg
def plot_accuracy_graph(history, save_path='accuracy_plot.jpeg'):
    """
    Plots and saves the training and validation accuracy graph.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path, format='jpeg', dpi=300)
    plt.show()

# Call the function to plot and save the accuracy graph
plot_accuracy_graph(history, save_path='training_validation_accuracy.jpeg')

# Plotting the training and validation loss graph and save as jpeg
def plot_loss_graph(history, save_path='loss_plot.jpeg'):
    """
    Plots and saves the training and validation loss graph.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(save_path, format='jpeg', dpi=300)
    plt.show()

# Call the function to plot and save the loss graph
plot_loss_graph(history, save_path='training_validation_loss.jpeg')
