# MobileNet-SSD Implementation

This project implements the MobileNet-SSD (Single Shot MultiBox Detector) model, which is optimized for object detection on mobile and embedded devices. The model balances speed and accuracy, enabling real-time detection and classification of multiple objects in images.

## Project Overview

- **Objective**: Implement an efficient object detection network for real-time applications on resource-constrained devices.
- **Model**: MobileNet-SSD, known for its lightweight architecture and ability to perform well on mobile platforms.
- **Tools**: Python, Caffe framework.
## Features

- **MobileNet Backbone**: Efficient convolutional layers to extract rich features from images.
- **SSD Head**: Detection layers that predict object locations and class scores.
- **Real-time Detection**: Capable of detecting multiple objects in real-time with minimal computational resources.


## File Structure

- `main.py`: Script for loading the MobileNet-SSD model and running the detection pipeline.
- `MobileNetSSD.txt`: Configuration file defining the layers of the MobileNet-SSD network.
- `MobileNetSSD_deploy.caffemodel`: Pre-trained weights for the MobileNet-SSD model.


## Getting Started

### Prerequisites

- Python 3.x
- Caffe framework
- Required Python packages: `numpy`, `opencv-python`, etc.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/MobileNet-SSD.git
   cd MobileNet-SSD

2. Download the pre-trained model:
Place the MobileNetSSD_deploy.caffemodel in the project directory.


## Example Output - 1
These results are the prediction from the input images

![Screenshot 2023-12-24 235527](https://github.com/user-attachments/assets/472a5046-0a64-47f2-80a3-f4b1944bafc2)


![Screenshot 2023-12-28 004605](https://github.com/user-attachments/assets/b58f012e-ff42-43d4-9cdc-f4003405199c)

![Screenshot 2023-12-28 004006](https://github.com/user-attachments/assets/8a29ba88-8c3a-4591-bd86-928617870f2f)

![Screenshot 2023-12-28 005319](https://github.com/user-attachments/assets/1d977256-b7c4-47bb-b5ec-70a66d5ee533)

## Actual Output - 2

This will be the main part of the project 

![Screenshot 2023-12-28 144038](https://github.com/user-attachments/assets/5179636d-23b4-4f3d-85b1-2e4f0c769cd2)
The above image segments the path need to be travelled from the environment which will be helpful for the car to predict the movement.

![Screenshot 2023-12-28 140141](https://github.com/user-attachments/assets/cc107ec2-cbce-460c-b0c5-14ebdd600ae6)

```

