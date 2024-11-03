ASL Letters Recognition
This project aims to recognize American Sign Language (ASL) letters using deep learning. A convolutional neural network (CNN) model is trained to classify images of ASL letters, allowing for recognition of hand gestures that represent each letter in the ASL alphabet.

Dataset
The ASL letters dataset is used to train and validate the model. It contains images labeled for each letter in the ASL alphabet. The dataset was formatted as TFRecord files for compatibility with TensorFlow's data processing.

Requirements
Python 3.9
TensorFlow 2.16 (or compatible version)
Other dependencies specified in requirements.txt

Training the Model
The model training script is located at src/model/train_asl_model.py. This script loads the ASL letters dataset, sets up a CNN model, and trains it on the training data.

Expected Output
The script will:

Load the dataset from the specified folders.
Train the CNN model on ASL letters data.
Output the training and validation accuracy.
Save the trained model as asl_model.pkl in the root directory.

Model Details
The model is a convolutional neural network (CNN) built using TensorFlow. It utilizes a ResNet-34 architecture, which is pre-trained on ImageNet, as a base model for transfer learning to improve classification accuracy on the ASL letters dataset.

