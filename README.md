# Smile-and-Sign-Recognition-CNN
📘 **Convolutional Model for Image Classification**

This repository contains a deep learning project built using TensorFlow and Keras Functional API. The notebook implements two image classification applications:

-Smile Detection on a custom dataset.

-Sign Language Recognition using a CNN model trained from scratch.

🧠 **Project Highlights**

📚 Learned Concepts: CNN layers (Conv2D, ReLU, MaxPooling, Flatten, Dense), Functional API vs Sequential API, data preprocessing, training with model.fit(), evaluation with model.evaluate().

🛠️ **Architecture:**

CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

🔍 **Optimizations**

-Categorical Crossentropy loss function

-Adam optimizer

-Image normalization

-Batch training using tf.data.Dataset

🧪 **Dataset**

The model uses the Happy House dataset and a custom Sign Language dataset. The datasets are processed using utility functions (cnn_utils.py) that perform:

-Loading .h5 dataset files

-Normalization of pixel values

-Conversion of labels to one-hot format

🧰**Tech Stack**

Language: Python 3

Libraries: TensorFlow, NumPy, Matplotlib, h5py

Model Type: Convolutional Neural Network (CNN)

Training Interface: Keras Functional API

🚀 **How to Run**

1. Clone the repo:
   
   git clone https://github.com/yourusername/Convolution_Model_Application.git
   
   cd Convolution_Model_Application
   
3. Install dependencies:
   
   pip install -r requirements.txt

5. Run the notebook: Open Convolution_model_Application.ipynb using Jupyter Notebook or Google Colab.

6. 📈 **Results**
   
Smile Detection Model: ✅ Works well with binary classification (happy/sad)

Sign Language Model: 📊 Achieved ~80% test accuracy

🧾 **Acknowledgements**

This notebook is part of the Deep Learning Specialization by Andrew Ng on Coursera.


