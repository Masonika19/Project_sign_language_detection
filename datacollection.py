import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
folder="Data_final/Good_luck"
counter=0
offset = 20
imgSize = 300


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

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
        else:
            # Wide image → width = imgSize, adjust height
            k = imgSize / w
            hCal = int(h * k)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        cv2.imshow('imgcrop', imgCrop)
        cv2.imshow('imgwhite', imgWhite)

    cv2.imshow('img', img)
    key=cv2.waitKey(1)
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)


