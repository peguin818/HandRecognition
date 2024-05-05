# ASL Recognition Mobile Application

## Table of Contents

1. [Project Overview](#project-overview)
2. [Final Product Images](#final-product-images)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
5. [Data](#data)
6. [Application Design](#application-design)
7. [Model Tuning](#model-tuning)
8. [How To Use](#how-to-use)

## Project Overview

The goal of this project is to empower deaf people by bridging communication gaps via the creation of an intuitive smartphone application that recognizes the alphabets of American Sign Language (ASL). This project makes use of deep learning and computer vision techniques to develop a mobile application that promotes more inclusive communication between hearing and deaf people.

## Final Product Images

### Home screen

This is the home screen of the application. From here, users can click on 'Start' button to start the prediction process.
Users must give camera permission in order for the application to work.

![image](https://github.com/peguin818/HandRecognition/assets/109569292/c8af5122-fb12-4a1b-9987-d493946ecb49)
![image](https://github.com/peguin818/HandRecognition/assets/109569292/221f1a79-ee52-4685-b8f4-04fb7993028d)

### Main screen

This is the main screen of the application. Users can perform ASL alphabet pose for the application to predict.
Users can choose to add the predicted letter to make phrases, add space between words, or delete the last added character.

![image](https://github.com/peguin818/HandRecognition/assets/109569292/1fab07cd-c1bc-4d4a-8969-d6e7045a4bbe)
![image](https://github.com/peguin818/HandRecognition/assets/109569292/ce3c67e5-dddc-479b-a495-bfc6ba58e942)

## Technology Stack

This project is built using the following technologies:

- Java: The primary programming language used for developing the application.
- Gradle: The build automation tool used for managing dependencies and building the project.
- TensorFlow: A popular open-source platform used for machine learning and artificial intelligence projects. It is used in this project to train a computer vision model for sign detection.

## System Architecture

The application trains a computer vision model specifically for mobile devices to provide efficient and real-time sign detection. A convolutional neural network (CNN) architecture created especially for mobile deployment serves as the system's fundamental component. This custom model is based on EfficientNetB7, which was selected for its accuracy and efficiency balance.

## Data

A publicly accessible dataset of labeled ASL alphabet hand pictures or videos will be used to train the algorithm.
Dataset link: https://www.kaggle.com/datasets/pramod722445/sign-language-dataset

## Application Design

The smartphone software will have an easy-to-use design when it is being developed, with an emphasis on Android. It will allow users to take pictures of hands signals using the device's camera. After that, the specially trained model receives the recorded video frames and processes them in real time.

## Model Tuning

The model will be highly precise and tuned for mobile resource restrictions to guarantee robustness. This balance will be attained by carefully adjusting the hyperparameters, which govern the model's learning process. In order to handle possible problems such as differences in lighting conditions or hand location, the trained model will be linked with the mobile application. The final accuracy of current model is about 80%.

## How to Use

### Prerequisites

- Java Development Kit (JDK) installed on your system. In this case, Amazon Corretto 8 (https://github.com/corretto/corretto-8)
- Gradle build tool installed on your system.
- An Android device or emulator for testing the application.

### Installation

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run `gradle build` to build the project.

### Setup

1. Open the project in your preferred IDE.
2. Sync the project with Gradle Files.
3. Run the application on your Android device or emulator.

### Usage

1. Open the application on your device.
2. Grant the necessary permissions for the application to access your device's camera.
3. Point your device's camera to the hand sign you want to recognize.
4. The application will process the image and display the recognized ASL alphabet on the screen.
