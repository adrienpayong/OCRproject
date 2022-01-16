# BUSINESS OVERVIEW:
People have been relying on paper invoices for a very long time but these days, all
have become digital and so are invoices. Reconciling digital invoices is currently a man-made
job but this can be automated to avoid men spending hours on browsing through several
invoices and noting things down in a ledger. This project deals with detecting three
important classes from the invoices. They are:
1. Invoice number
2. Billing Date
3. Total amount

## step-by-step guide on building OCR from scratch in Python:

This project is applicable to any field that is currently manually going through all of the bills to jot them down in a ledger. 

## TECH STACK:

- Language: Python
- Object detection : YOLO V4
- Text Recognition : Tesseract OCR
- Environment: Google Colab

## APPROACH:

**1. Setting up and Installation to run Yolov4**

We'll be downloading AlexeyAB's famous repository, adjusting the Makefile to enable OPENCV and GPU for darknet, and then building darknet. 

**2. Downloading pre-trained YOLOv4 weights**

YOLOv4 has already been trained on the coco dataset, which contains 80 classes that it can predict.
We'll use these pretrained weights to see how they perform on some of the images. 

**3. Creating display functions to display the predicted class**
**4. Data collection and Labeling with LabelIm**
To build a custom object detector, we'll need a large dataset of images and labels to train the model, so we'll use annotation tool LabelImg  to label our images.

**5. Configuring Files for Training**

This step involves configuring custom .cfg, obj.data, obj.names, train.txt and
test.txt files.

  a. Configuring all the needed variables based on class in the config file
  
  b. Creating obj.names and obj.data files
  
  c. Configuring train.txt and test.txt
  
**6. Download pre-trained weights for the convolutional layers**

**7. Training Custom Object Detector**

**8. Evaluating the model using Mean Average precision**

**9. Predict image classes and save the co-ordinates separately**

**10. Detecting text from the predicted class**

   - Importing pytesseract and setting environment variable for english trained
data
   - Getting list of predicted files from the director
   - Using tesseract pretrained LSTM model to extract the text
   - Fine tuning the LSTM model

