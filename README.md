# BUSINESS OVERVIEW
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
  
  Create a new file called obj.names where you will have one class name per line in the same order as your classes.txt from the dataset generation step.
  
  ![source](https://github.com/adrienpayong/OCRproject/blob/main/Captureinv.PNG)
  
  You will also need to create a obj.data file and fill it in like this(change your class number accordingly.
  This backup path is where we will save the weights to of our model throughout training. Upload obj.names and obj.data in data subfolder of darknet
  
   ![source](https://github.com/adrienpayong/OCRproject/blob/main/Captureinv1.PNG)
  
  c. Configuring train.txt and test.txt
  
  The last configuration files needed before we can begin to train our custom detector are the train.txt and test.txt files which will have paths to all our training images and valdidation images. To create the same we will be using two script for generating train and text text files.
  
  ```
  !python /content/generate_train.py
!python /content/generate_test.py
```
**6. Download pre-trained weights for the convolutional layers**

This step downloads the weights for the convolutional layers of the YOLOv4 network. By using these weights it helps your custom object detector to be way more accurate and not have to train as long.
```
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

**7. Training Custom Object Detector**

We are now ready to train our custom YOLOv4 object detector on Amount,Invoice Number and Date classes. So run the following command.( -dont_show flag stops chart from popping up since Colab Notebook can't open images on the spot, -map flag overlays mean average precision on chart to see the accuracy)
```
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map
```

If for some reason you get an error or your Colab goes idle during training, you have not lost your partially trained model and weights because every 100 iterations a weights file called yolov4-obj_last.weights is saved to content/backup/ folder.
We can kick off training from our last saved weights file so that we don't have to restart!
Just run the following command but with your backup location.

```
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg /content/backup/yolov4-obj_last.weights -dont_show
```

**8. Evaluating the model using Mean Average precision**

If you didn't run the training with the '-map- flag added then you can still find out the mAP of your model after training. Run the following command on any of the saved weights from the training to see the mAP value for that specific weight's file.
```
!./darknet detector map data/obj.data cfg/yolov4-obj.cfg /content/backup/yolov4-obj_last.weights
```

**9. Predict image classes and save the co-ordinates separately**

Run your custom detector with this command
```
!./darknet detector test data/obj.data cfg/yolov4-obj.cfg /content/drive/MyDrive/yolov4-obj_best.weights /content/images/basic-invoice.png -thresh 0.4
imShow('predictions.jpg')
```

You will get this at the end:

![source](https://github.com/adrienpayong/OCRproject/blob/main/Capturefact.PNG)


**10. Detecting text from the predicted class**

   - Importing pytesseract and setting environment variable for english trained data
   
```
import pytesseract as p
import os
os.environ["TESSDATA_PREFIX"] = 'C:/Users/being/Downloads/Invoice_pro/tesseract_training/Tesseract-OCR/tessdata/'
```

   - Getting list of predicted files from the director

![source](https://github.com/adrienpayong/OCRproject/blob/main/extract.PNG)

   - Using tesseract pretrained LSTM model to extract the text

![source](https://github.com/adrienpayong/OCRproject/blob/main/Captureloop.PNG)

   - Fine tuning the LSTM model
   
     ![source](https://github.com/adrienpayong/OCRproject/blob/main/Capturefine.PNG)
     
    - Link: ![Class detection invoice](https://github.com/adrienpayong/OCRproject/blob/main/CLASS_DETECTION_INVOICE.ipynb)
    - Link1: ![image to text conversion](https://github.com/adrienpayong/OCRproject/blob/main/Image_to_Text_Conversion(Tesseract).ipynb)

