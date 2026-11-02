# Cat-vs-Dog-Image-Classification-Using-Deep-Learning-MobileNetV3-Streamlit
A deep learning-based image classification project that uses a fine-tuned MobileNetV3-Small model with PyTorch to identify cats and dogs. Built with Streamlit, it provides an interactive interface for uploading images and displaying real-time predictions with confidence scores.

# 🚀 Key Features
- **Image Classification**: Classifies uploaded images into two categories: Cat or Dog.
- **Deep Learning Model**: Uses MobileNetV3-Small with a modified classifier layer for binary classification.
- **Confidence Score:** Displays the model's prediction confidence as a percentage.
- **Confidence Threshold:** Uses a 70% confidence threshold to avoid displaying uncertain predictions.
- **Interactive Web Interface:** Built with Streamlit for easy image uploading and prediction.
- **GPU Support:** Automatically detects and uses CUDA when available, otherwise runs on CPU.
- **Image Preprocessing:** Applies resizing, center cropping, tensor conversion, and ImageNet normalization.
# Technologies Used
- Python
- PyTorch
- Torchvision
- Streamlit
- PIL (Pillow)
- Deep Learning
- CNN
- Transfer Learning (MobileNetV3-Small)
# How It Works
- Upload a cat or dog image through the Streamlit application.
- The uploaded image is preprocessed using the required transformations.
- The trained MobileNetV3-Small model analyzes the image.
- Softmax converts model outputs into class probabilities.
- The application displays the predicted class and confidence score.
- If confidence is below 70%, the application indicates that the model is not confident enough to classify the image.
# 🎯 Project Objective
The main objective of this project is to demonstrate how deep learning and transfer learning can be used to build an image classification system with an interactive web application.
