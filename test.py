import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from tensorflow.keras.models import load_model
import numpy as np


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = load_model("D:\projects_pycharm\model_new\Data_collection_sign_model.h5")
#folder="Data/Yes"
counter=0

offset = 20
imgSize = 300

labels =["Call","Good_luck","Hello","I_love_you","Okay","Peace","please","Thank_you","Thumbs_up","Yes"]

while True:
    success, img = cap.read()
    imgOutput =img.copy()
    hands,img = detector.findHands(img,draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        height, width = img.shape[:2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Safely calculate crop area
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(width, x + w + offset)
        y2 = min(height, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        aspectRatio = h / w

        if aspectRatio > 1:
            # Tall image → height = imgSize, adjust width
               k = imgSize / h
               wCal = int(w * k)
               imgResize = cv2.resize(imgCrop, (wCal, imgSize))
               wGap = (imgSize - wCal) // 2
               imgWhite[:, wGap:wGap + wCal] = imgResize

               imgWhiteResized = cv2.resize(imgWhite, (300, 300))           # Step 1: Resize
               imgInput = imgWhiteResized.astype(np.float32)/255.0
               imgInput=np.expand_dims(imgInput,axis=0)# Step 2: Flatten
               prediction = model.predict(imgInput)

               index = np.argmax(prediction)
               print(prediction,index)
               print("Confidence:", np.max(prediction))

        else:
            # Wide image → width = imgSize, adjust height
              k = imgSize / w
              hCal = int(h * k)
              imgResize = cv2.resize(imgCrop, (imgSize, hCal))
              hGap = (imgSize - hCal) // 2
              imgWhite[hGap:hGap + hCal, :] = imgResize
              imgWhiteResized = cv2.resize(imgWhite, (300, 300))  # Step 1: Resize
              imgInput = imgWhiteResized.astype(np.float32)/255.0
              imgInput=np.expand_dims(imgInput,axis=0)# Step 2: Flatten
              prediction = model.predict(imgInput)
              index=np.argmax(prediction)
              print(prediction, index)



        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x-offset+90, y-offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
        cv2.imshow('imgcrop', imgCrop)
        cv2.imshow('imgWhite', imgWhite)

        cv2.imshow('img', imgOutput)
        cv2.waitKey(1)


