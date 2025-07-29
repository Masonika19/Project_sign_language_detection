# Project_sign_language_detection
#Realtime_sign_language_detection using CNN


Sign language is a crucial means of communication for individuals with hearing and speech impairments. However, most people are not familiar with sign language, creating a communication barrier. This project aims to bridge that gap by developing a real-time sign language detection system using deep learning.

In this project, a Convolutional Neural Network (CNN) model is trained to recognize 10 distinct hand signs captured through images. These signs include commonly used gestures such as:

Hello, Thank You, I Love You, Yes, Please, Okay, Call, Good Luck, Peace, and Thumbs Up.

Each sign class contains over 300 images, manually collected to build a custom dataset. These images were resized to 224x224 pixels for efficient training and performance. The CNN model learns the visual features of each sign and is able to classify them accurately in real time.

The model was trained from scratch using the TensorFlow/Keras framework, incorporating techniques like data augmentation, normalization, and early stopping to improve generalization and prevent overfitting. The dataset was split into training, validation, and testing sets to evaluate the model's performance.

Once trained, the model was integrated into a real-time prediction system that processes webcam input, identifies the hand sign, and displays the predicted label.
