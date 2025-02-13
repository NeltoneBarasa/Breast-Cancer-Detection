#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import os
import glob
import pandas as pd             # Pandas for data manipulation
import numpy as np              # NumPy for numerical operations
import matplotlib.pyplot as plt # Matplotlib for visualization
import seaborn as sns           # Seaborn for plotting
from PIL import Image           # Pillow for image processing

# TensorFlow and Keras for building deep learning models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.applications import DenseNet121


# # Extracting and Preparing the Dataset for Training

# In[2]:


import zipfile
import os

# Define the path to the zipped dataset and the extraction directory
zip_path = r"C:\Users\USER\OneDrive\Desktop\archive (15).zip"
extraction_path = r"C:\Users\USER\OneDrive\Desktop\breast_ultrasound_dataset"

# Extract the dataset if it hasn't been extracted
if not os.path.exists(extraction_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

# Define the training data directory after extraction
train_data = os.path.join(extraction_path, 'Dataset_BUSI_with_GT')


# # Organizing and Labeling the Dataset for Training

# In[3]:


# List files in the training data directory
file_names = os.listdir(train_data)

# Create a DataFrame with file names
df = pd.DataFrame(file_names, columns=['File Name'])
print(df.head())

# Collect file paths and labels
train_files = [i for i in glob.glob(train_data + "/*/*")]
np.random.shuffle(train_files)  # Randomly shuffle file paths

labels = [os.path.dirname(i).split("/")[-1] for i in train_files]  # Extract labels from the directory structure

# Combine file paths & labels into a DataFrame
data = zip(train_files, labels)
training_data = pd.DataFrame(data, columns=["Path", "Label"])

# Display the DataFrame
print(training_data.head())


# # Label Distribution Visualization

# In[4]:


# Visualizing label distribution using Seaborn
ax = sns.countplot(x=training_data["Label"])

# Add counts inside bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)

plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.title(f'Total Files: {len(training_data)}', pad=10)
plt.show()


# # Data Preprocessing and Splitting into Training, Validation, and Test Sets

# In[5]:


from sklearn.model_selection import train_test_split



# Set the batch size for training
batch_size = 16
image_size = (256, 256)  # Define target image size (256x256)
num_channels = 3  # RGB images

# Define preprocessing function (you can customize this function later if necessary)
def preprocess_image(img):
    return img

# Split the dataset into training, validation, and test sets
train_df, val_test_df = train_test_split(training_data, train_size=0.8, shuffle=True, random_state=123)
val_df, test_df = train_test_split(val_test_df, train_size=0.5, shuffle=True, random_state=123)

# Define ImageDataGenerator for real-time data augmentation
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image, 
    rescale=1.0/255
)

# Convert the 'Label' column to string type
train_df['Label'] = train_df['Label'].astype(str)
val_df['Label'] = val_df['Label'].astype(str)

# Create generators for training, validation, and testing
train_generator = datagen.flow_from_dataframe(
    train_df, 
    x_col='Path', 
    y_col='Label',
    target_size=image_size, 
    class_mode='categorical',
    color_mode='rgb', 
    shuffle=True, 
    batch_size=batch_size
)

valid_generator = datagen.flow_from_dataframe(
    val_df, 
    x_col='Path', 
    y_col='Label',
    target_size=image_size, 
    class_mode='categorical',
    color_mode='rgb', 
    shuffle=True, 
    batch_size=batch_size
)


# # Extracting and Displaying Class Labels and Indices

# In[6]:


# Get class indices from the training data generator
class_indices = train_generator.class_indices

# Print out the class names and their total count
labels = list(class_indices.keys())
total_labels = len(labels)

print("Labels:", labels)
print("\nTotal number of unique labels:", total_labels)


# # Visualizing a Subset of Training Images with Labels

# In[8]:


# Display a subset of training images
no_of_images = 8  # Display 8 images
fig, axes = plt.subplots(no_of_images, 1, figsize=(5, no_of_images * 2))  # Adjust size for vertical layout

for i in range(no_of_images):
    index = i  # Accessing image index
    if index < len(training_data):
        img = Image.open(training_data.iloc[index]['Path'])
        axes[i].imshow(img)
        axes[i].axis('off')  # Hide axis for clean visualization
        label = training_data.iloc[index]['Label']
        axes[i].set_title(label, fontsize=12, pad=10)

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()


# # Building a Custom CNN Model on DenseNet121 for Classification

# In[7]:


# Load DenseNet121 model without the top layers (include_top=False)
base_model = DenseNet121(
    weights='imagenet',  # Use pre-trained ImageNet weights
    include_top=False,   # Exclude the fully connected layers
    input_shape=(256, 256, 3)  # Adjust input size to match images (256x256)
)

# Freeze the layers of the pre-trained model to prevent updates during training
for layer in base_model.layers:
    layer.trainable = False

# Build the custom model on top of DenseNet121
model = Sequential()
model.add(base_model)
model.add(Flatten())  # Flatten the output of DenseNet121
model.add(Dense(1024, activation='relu'))  # First fully connected layer
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(1024, activation='relu'))  # Second fully connected layer
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(512, activation='relu'))  # Third fully connected layer
model.add(Dense(128, activation='relu'))  # Fourth fully connected layer
model.add(Dense(len(labels), activation='softmax'))  # Output layer with the number of classes

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()


# # Training the Custom DenseNet121 Model

# In[8]:


# Define the number of epochs and steps per epoch
epochs = 10  # Adjust as needed
steps_per_epoch = len(train_generator)  # Number of batches per epoch

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)
)


# # Model Evaluation

# In[9]:


# Evaluate the model on the test set
test_generator = datagen.flow_from_dataframe(test_df, x_col='Path', y_col='Label', 
                                             target_size=image_size, class_mode='categorical', 
                                             batch_size=batch_size)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")


# # Evaluation on Training and Validation Sets

# In[10]:


# Evaluate the model on the train set
train_loss, train_accuracy = model.evaluate(train_generator, steps=train_generator.samples // train_generator.batch_size)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(valid_generator, steps=valid_generator.samples // valid_generator.batch_size)

# Convert accuracy to percentage
train_accuracy_percentage = train_accuracy * 100
val_accuracy_percentage = val_accuracy * 100

# Create a Pandas DataFrame to display the results
evaluation_results = pd.DataFrame({
    'Set': ['Train', 'Validation'],
    'Loss': [train_loss, val_loss],
    'Accuracy': [f'{train_accuracy_percentage:.2f}%', f'{val_accuracy_percentage:.2f}%']
})

# Display the evaluation results DataFrame
evaluation_results


# # Training and Validation Loss Visualization

# In[11]:


import matplotlib.pyplot as plt

# Assuming 'history' is the object returned from model.fit()

# Plotting the training and validation loss
plt.figure(figsize=(12, 6))

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Display the plot
plt.show()


# # Confusion Matrix

# In[12]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(model, test_generator, class_labels):
    
    # Get true labels and predictions
    true_labels = test_generator.classes  # True labels from test generator
    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class indices

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='viridis', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

# Example usage:
plot_confusion_matrix(model, test_generator, labels)


# # Classification Report

# In[14]:


from sklearn.metrics import classification_report
import numpy as np

def generate_classification_report(model, test_generator, class_labels):
    
    # Get true labels and predictions
    true_labels = test_generator.classes  # True labels from test generator
    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class indices

    # Generate classification report
    report = classification_report(true_labels, predicted_labels, target_names=class_labels)
    print("Classification Report:\n")
    print(report)

# Example usage:
generate_classification_report(model, test_generator, labels)


# # Making Predictions (Classify Images)

# In[17]:


import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

def classify_and_display_with_true_labels(model, dataframe, class_labels, target_size=(256, 256), num_images=5):
    """
    Classifying images and display them alongside true labels and predictions.

    Args:
        model: Trained Keras model.
        dataframe (pd.DataFrame): DataFrame containing image paths and true labels.
        class_labels (list): List of class labels corresponding to model output.
        target_size (tuple): Target size for resizing images (default is (256, 256)).
        num_images (int): Number of images to display (default is 5).
    """
    plt.figure(figsize=(12, num_images * 3))  # Adjust figure size dynamically

    # Limit the number of images to display
    displayed_images = dataframe.sample(n=num_images, random_state=123).reset_index()  # Randomly sample images

    for i, row in displayed_images.iterrows():
        img_path = row['Path']
        true_label = row['Label']

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)  # Get index of the highest probability
        predicted_label = class_labels[predicted_index]
        confidence = predictions[0][predicted_index]

        # Display the image and prediction
        plt.subplot(num_images, 1, i + 1)  # Create subplots for each image
        plt.imshow(img)
        plt.axis('off')  # Turn off the axes for cleaner visualization
        plt.title(
            f"True: {true_label} | Predicted: {predicted_label} ({confidence * 100:.2f}%)",
            fontsize=14, 
            color="blue" if true_label == predicted_label else "red"
        )

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

# Example usage:
classify_and_display_with_true_labels(model, training_data, labels, num_images=5)


# In[ ]:




