

# Brain Tumor Classification using Convolutional Neural Network (CNN)

This repository contains a Python script for classifying brain tumor images using a Convolutional Neural Network (CNN). The model is trained to distinguish between two categories: "no tumor" and "yes tumor."

## Table of Contents
- [Project Overview](#project-overview)
- [Libraries Used](#libraries-used)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Prediction](#prediction)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to build and train a CNN model to classify brain tumor images. The model is trained on a dataset containing images labeled as "no tumor" and "yes tumor." The trained model can then be used to make predictions on new images.

## Libraries Used
- **cv2 (OpenCV)**: For image processing.
- **os**: For handling file and directory paths.
- **tensorflow and keras**: For building and training the deep learning model.
- **PIL (Pillow)**: For image handling.
- **numpy**: For numerical operations.
- **sklearn.model_selection**: For splitting the dataset into training and testing sets.

## Data Preparation
### Directory Structure
The data should be organized into the following directory structure:
```
image_directory/
├── no/
│   ├── no0.jpg
│   ├── no1.jpg
│   └── ...
├── yes/
│   ├── yes0.jpg
│   ├── yes1.jpg
│   └── ...
```

### Loading and Resizing Images
1. Iterate over the images in both "no" and "yes" directories.
2. Check if the file extension is `.jpg`.
3. Load the image using `cv2`, convert it to RGB format using `PIL`, and resize it to `(64, 64)`.
4. Append the resized image and its label (0 for "no tumor", 1 for "yes tumor") to the dataset and label lists.

### Conversion to NumPy Arrays
Convert the dataset and label lists to NumPy arrays for further processing.

### Data Splitting
Split the dataset into training and testing sets using `train_test_split` with 20% of the data reserved for testing.

### Normalization
Normalize pixel values of images using `normalize` from Keras to scale pixel values to the range `[0, 1]`.

### One-Hot Encoding
Convert labels to one-hot encoded format using `to_categorical`.

## Model Architecture
- **Sequential model**: A linear stack of layers.
- **Convolutional layers**:
  - `Conv2D` with 32 filters, kernel size `(3,3)`, and ReLU activation.
  - `MaxPooling2D` with pool size `(2,2)`.
  - Additional `Conv2D` layers with 32 and 64 filters, and `MaxPooling2D`.
- **Flatten the output** and add fully connected (Dense) layers:
  - `Dense` layer with 64 units and ReLU activation.
  - `Dropout` layer with 50% rate to prevent overfitting.
  - Final `Dense` layer with 2 units (one for each class) and softmax activation.

## Training
- **Loss function**: `categorical_crossentropy` (suitable for multi-class classification).
- **Optimizer**: `adam`.
- **Metrics**: Accuracy.
- **Batch size**: 16
- **Number of epochs**: 10
- **Validation data**: Testing set
- **Shuffle**: False (to maintain the order of data)

## Prediction
### Loading the Model
Load the pre-trained model using `load_model` from Keras.

### Preprocessing the Image
1. Load the image using `cv2.imread`.
2. Check if the image was loaded correctly. Raise an error if the image file is not found.
3. Convert the image to RGB format using `PIL`, resize it to `(64, 64)`, and convert it to a NumPy array.
4. Expand dimensions of the image array to match the input shape expected by the model using `np.expand_dims`.

### Making a Prediction
Use the `predict` method of the model to get the prediction for the preprocessed image. Determine the class with the highest probability using `np.argmax` on the prediction array.

### Example Output
The script will print the predicted class for the given image. The output will be either `0` or `1`, corresponding to "no tumor" and "yes tumor," respectively.

## Usage
### Training the Model
1. Clone the repository.
2. Install the required dependencies.
3. Organize your data into the specified directory structure.
4. Run the training script to preprocess data, build the model, and train it.

### Making Predictions
1. Ensure the trained model file (`BrainTumor10EpochsCategorical.h5`) is available.
2. Run the prediction script to load the model and make predictions on new images.

## Contributing
Contributions to this project are welcome. Please open an issue or submit a pull request with your changes.

## License
This project is licensed under the MIT License.
