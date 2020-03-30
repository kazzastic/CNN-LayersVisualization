#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 02:16:54 2020

@author: kazzastic
"""

from anything.gradcam import GradCam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
#from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import argparse
import imutils
import cv2


def prepare(filepath):
    IMG_SIZE = 220 # 50 in txt-based
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--model", type=str, default="vgg", choices=("vgg", "resnet"), help="model to be used")
args = vars(ap.parse_args())

Model = VGG16

if args["model"] == "resnet":
    Model = ResNet50

print("[INFO] Loading Model...")
model = load_model("NIC-CNN.model")

orig = cv2.imread(args["image"])
#resized = cv2.resize(orig, (224, 224))

#image = load_img(args["image"], target_size=(224, 224))
#image = img_to_array(image)
#image = np.expand_dims(image, axis=0)
#image = preprocess_input(image)
image = prepare(args["image"])
preds = model.predict(image)
i = np.argmax(preds[0])

#decoded = decode_predictions(preds)
#(imagenetID, label, prob) = decoded[0][0]
classes = ["WHITE-HOUSE", "NIC"]
label = classes[int(preds[0][0])]
label = "{}:".format(label)
print("[INFO] {}".format(label))

cam = GradCam(model, i)
heatmap = cam.compute_heatmap(image)

heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)


cv2.rectangle(output, (0,0), (340, 40), (0,0,0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)