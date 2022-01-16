# BUSINESS OVERVIEW:
People have relied on paper invoices for a long time, but everything has gone digital, including invoices.
Reconciling digital invoices is currently a man-made task, but it can be automated to save men from spending hours scrolling through multiple invoices and noting things down in a ledger. This project deals with detecting three
important classes from the invoices. They are:
1. Invoice number
2. Billing Date
3. Total amount

## APPROACH:

This project is applicable to any field that is currently manually going through all of the bills to jot them down in a ledger. 

## TECH STACK:

- Language: Python
- Object detection : YOLO V4
- Text Recognition : Tesseract OCR
- Environment: Google Colab

## Topics Covered in this Project

All of the data science tools and techniques that you will use to implement this project's solution are detailed below. 

## step-by-step guide on building OCR from scratch in Python:

**1. Setting up and Installation to run Yolov4**

We'll be downloading AlexeyAB's famous repository, adjusting the Makefile to enable OPENCV and GPU for darknet, and then building darknet. 

## Computer Vision using OpenCV

OpenCV is one of Python's most popular image processing and computer vision libraries.
Any image must be processed before being served to the object detection model YOLO, and you will use OpenCV to do so.
Furthermore, various OpenCV library functions are used to visualize the YOLO model's testing results. 

## Text Detection using YOLO

YOLO v4 is an object detection model created by Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao, with the acronym YOLO standing for 'You Only Look Once.'
Its humorous name derives from the algorithm's ability to identify all of the objects in an image with a single glance.
That is one of the main reasons why the algorithm detects objects faster than RCNN methods.
The YOLO technique will be used in this project to create a custom OCR using Python.
The motivation for developing a custom OCR model is that YOLO can only detect 80 preset classes in the COCO dataset.
As a result, this project will walk you through the transfer learning process of developing a YOLO-text-recognition model using the invoices dataset.
As previously said, our custom OCR system will recognize three objects from invoice images: invoice number, Billing Date, and Total amount. 

## OCR using Tessaract

The system will recognize the important text classes from the invoices using YOLO, but to decode the information in the text, Optical Character Recognition must be used (OCR).
Tessaract OCR is a tool that scans text and converts it to digital data quickly.
You will learn how to use Tessaract OCR to create a custom OCR in Python in this project. 

## Coding with Google Colab

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

