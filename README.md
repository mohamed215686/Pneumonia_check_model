# Pneumonia Detection from Chest X-Rays 🫁

This project is a Deep Learning model built to automatically detect and classify pneumonia from chest X-ray images. It uses a Convolutional Neural Network (CNN) trained on a balanced dataset of medical images to distinguish between healthy lungs and lungs infected with pneumonia.

## 🛠️ Technologies Used
* **Python:** The core programming language.
* **TensorFlow / Keras:** Used to build, train, and save the Deep Learning model.
* **OpenCV (cv2):** Used for image processing and validating the integrity of the image files.
* **Matplotlib:** Used for visualizing the datasets and plotting the training/validation loss graphs.
* **Kaggle Notebooks:** Environment used for model development and GPU acceleration (NVIDIA T4/P100).

## 📊 The Dataset
The model is trained on a perfectly balanced structure of Chest X-Ray images to prevent class bias. The data is divided into three directories:
* **Train:** 6,800 images (3,400 NORMAL, 3,400 PNEUMONIA)
* **Validation:** 1,700 images (850 NORMAL, 850 PNEUMONIA)
* **Test:** 30 images (15 NORMAL, 15 PNEUMONIA)

*Note: All images are preprocessed and resized to 256x256 pixels before being fed into the network.*

## 🧠 Model Architecture & Techniques
The core of this project is a custom Convolutional Neural Network (CNN) designed for image classification. Key features of the training pipeline include:
1.  **Data Augmentation:** Random rotations, zooming, and flipping are applied to the training data to make the model more robust and prevent it from memorizing specific images.
2.  **Global Average Pooling:** Used in place of traditional Flatten layers to drastically reduce the number of parameters and combat overfitting.
3.  **Dropout Layers:** Randomly turns off a percentage of neurons during training to force the network to learn general, underlying patterns of the disease.
4.  **CPU/GPU Optimization:** Implemented `tf.data.AUTOTUNE` and `prefetching` to eliminate data bottlenecks and keep the GPU fully utilized during training.

## 🚀 How to Run
1. Clone this repository to your local machine or open it in a Kaggle Notebook.
2. Ensure you have the required libraries installed:
   ```bash
   pip install tensorflow opencv-python matplotlib
