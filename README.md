# BUSINESS OVERVIEW:
People have been relying on paper invoices for a very long time but these days, all
have become digital and so are invoices. Reconciling digital invoices is currently a man-made
job but this can be automated to avoid men spending hours on browsing through several
invoices and noting things down in a ledger. This project deals with detecting three
important classes from the invoices. They are:
1. Invoice number
2. Billing Date
3. Total amount

## APPLICATIONS:

This project is applicable to any field that is currently manually going through all of the bills to jot them down in a ledger. 

## TECH STACK:

- Language: Python
- Object detection : YOLO V4
- Text Recognition : Tesseract OCR
- Environment: Google Colab

## APPROACH:

1. Setting up and Installation to run Yolov4

We'll be downloading AlexeyAB's famous repository, adjusting the Makefile to enable OPENCV and GPU for darknet, and then building darknet. 

2. Downloading pre-trained YOLOv4 weights
3. 
YOLOv4 has already been trained on the coco dataset, which contains 80 classes that it can predict.
We'll use these pretrained weights to see how they perform on some of the images. 
