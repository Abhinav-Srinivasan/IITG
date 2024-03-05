

import os
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

data_path = r"D:\work\IITG\archive\animals\animals"

# Step 1: Organize the Dataset
# (Ensure each folder contains images of the corresponding animal)

# Step 2: Create Binary Labels
classes = sorted(os.listdir(data_path))
num_classes = len(classes)

labels = []
images = []

for class_idx, class_name in enumerate(classes):
    class_path = os.path.join(data_path, class_name)
    if os.path.isdir(class_path):
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            images.append(image_path)
            # Assign binary labels (1 for the current class, 0 for others)
            labels.append(class_idx)

# Step 3: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 4: Shuffle the Dataset
combined = list(zip(X_train, y_train))
random.shuffle(combined)
X_train[:], y_train[:] = zip(*combined)

combined = list(zip(X_test, y_test))
random.shuffle(combined)
X_test[:], y_test[:] = zip(*combined)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, labels):
    images = []
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        images.append(img_array)
    return np.array(images), np.array(labels)



# Step 4: Load and preprocess the training dataset
X_train, y_train = load_and_preprocess_images(X_train, y_train)

# Step 5: Load and preprocess the testing dataset
X_test, y_test = load_and_preprocess_images(X_test, y_test)

# Step 6: Train one model for each class using VGG16
models_dict = {}

for target_class in range(num_classes):
    # Create binary labels (1 for the target class, 0 for others)
    binary_labels = np.where(y_train == target_class, 1, 0)

    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification layer
    ])

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, binary_labels, epochs=10, batch_size=32, validation_split=0.2)

    # Save the model in a dictionary
    models_dict[target_class] = model

# Step 7: Evaluate the models on the test set
all_predictions = []

for target_class, model in models_dict.items():
    binary_test_labels = np.where(y_test == target_class, 1, 0)
    predictions = (model.predict(X_test) > 0.5).astype(int)
    all_predictions.append(predictions)

# Step 8: Combine the predictions
combined_predictions = np.vstack(all_predictions).T
final_predictions = np.argmax(combined_predictions, axis=1)

# Step 9: Calculate and print metrics
accuracy = accuracy_score(y_test, final_predictions)
print(f"Overall Accuracy: {accuracy}")

# Display detailed classification report
print(classification_report(y_test, final_predictions))




# Set your data path
data_path = r"D:\work\IITG\archive\animals\animals"


# Step 1: Organize the Dataset for 5 classes
selected_classes = ['bat', 'bee', 'cat', 'mouse', 'owl']

labels = []
images = []

for class_idx, class_name in enumerate(selected_classes):
    class_path = os.path.join(data_path, class_name)
    if os.path.isdir(class_path):
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            images.append(image_path)
            # Assign labels for the selected classes (0 to 4)
            labels.append(class_idx)

# Step 2: Split the Dataset
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(images, labels)):
    print(f"\nTraining Fold {fold + 1}")

    X_train, X_test = np.array(images)[train_index], np.array(images)[test_index]
    y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

    # Step 3: Load and preprocess the training dataset
    X_train, y_train = load_and_preprocess_images(X_train, y_train)

    # Step 4: Load and preprocess the testing dataset
    X_test, y_test = load_and_preprocess_images(X_test, y_test)

    # Step 5: Train a more sophisticated CNN model
    model = models.Sequential([
        layers.Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(len(selected_classes), activation='softmax')  # Multi-class classification layer
    ])

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Step 6: Evaluate the model on the test set
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Step 7: Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for Fold {fold + 1}: {accuracy}")

    # Display detailed classification report
    print(classification_report(y_test, y_pred))

