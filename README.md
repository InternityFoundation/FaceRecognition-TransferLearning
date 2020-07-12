# FaceRecognition-TransferLearning
The whole project follows 3 steps of creating a Machine Learning Code i.e -
a) Data Preparation
b) Model Training 
c) Model Prediction

# Data Preparation (Take my photos.ipynb)
Your system must have a webcam , installed Python 3.x and opencv-python to run the code.
If youwant to train the model to predict 5 people , then take 60 photos of each preson and save them in a particular folder .
Create a Training Dataset and Validation Dataset and put these two folders in a single folder .
Dataset is ready . Now upload this Dataset on Google Drive

# Model Training (Face_recog_Vgg16.ipynb)
   # Combining the models and printing the summary
Change the num_classes=5 as required  (if your dataset has 6 classes then change to num_classes=6)
   # Importing dataset and using Image generator for Augmentation
Change the train_data_dir and val_data_dir and specify the path of your dataset on the google drive.
   # Training the Model
Change the path where you want to save your model with model_name.h5 
nb_train_samples = (total no. of images in your train dataset)
nb_validation_samples = (total no. of images in your test dataset)

# Model Prediction (Vgg16_prediction.ipynb)
 classifier = load_model('path to model/model_name.h5')
 Update the face_dict and the face_dict_n with the names of your classes that you are predicting and the class names should be same as the folder names given in the prediction    set .
 input_im = getRandomImage("path of prediction folder")

# Model Training and Prediction(INTERNITY_face_detector_RAHUL.ipynb)
  # Training
  Upload the zip file of the training_data and validation data(testing_data) and unzip them in the code .
  Change the num_classes=5 as required  (if your dataset has 6 classes then change to num_classes=6)
  # Prediction 
  Change the path of prediction_data (data should have some images of each class in a separate folder put in a simgle folder)
  input_im = getRandomImage("path of prediction data folder")
  Update the face_dict and the face_dict_n with the names of your classes that you are predicting and the class names should be same as the folder names given in the prediction    set .
  
