# Python Face_Mask_Detection

# Import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (500, 500), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initialize our list of faces, their corresponding locations and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > args["confidence"]:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI(Region of Interest), convert it from BGR to RGB channel ordering,
            # resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # Add the face and bounding boxes to their respective lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # Only make a predictions if at least one face was detected
    if len(faces) == 1:
        # For faster inference we'll make batch predictions on *all* faces at the same time
        # rather than one-by-one predictions in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # Return a 2-tuple(Ordered Pair, For holding two values together of any data type) of the face locations
    # and their corresponding locations
    return locs, preds


# Construct the argument parser(To read the file containing the arguments) and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load our serialized face detector model from disk
# os.path.sep.join is a method used to join components of a file with the appropriate seperator for the current os
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])  # .prototxt defines the architecture and parameters
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])  # .caffemodel contains the learned
                                                                              # weights of the model
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)   # cv2.dnn.readNet is a function from OpenCV's dnn module used
                                                       # to read learning network models

# Load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("model.h5")

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to have a maximum width of 500 pixels
    # (Chose 400 pixels as a balance between reducing computational load while still maintaining enough detail
    # for the application's needs)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # Unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Determine the class label and color we will use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # # Include the probability in the label
        # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()

# # Observation:
# 1)Import the necessary packages
# 2)We are creating a blob and a screen for our face to be seen and highlighted
# 3)Next based on the above part we make predictions if atleast 1 face is seen
# 4)Read the file containing the arguments and parse the arguments like diving the data and running
# 5)Load the pre-trained model for training and solving the problems
# 6)Initialise the video screen and let it warm up for a time duration
# 7)Keep looping until the user clicks the key q or stops running the program
# 8)Keep a proper cleanup after running
