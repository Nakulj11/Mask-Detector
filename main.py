import os
import cv2
import numpy as np


def main():
    # cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    # dir = os.path.join(cv2_base_dir, '/modules/face/data/cascades')
    nose_cascade = cv2.CascadeClassifier("nose.xml")
    mouth_cascade = cv2.CascadeClassifier("mouth.xml")
    detector = cv2.CascadeClassifier("face.xml")

    if nose_cascade.empty() or mouth_cascade.empty():
        raise IOError('Unable to load the cascade classifier xml file')

    cap = cv2.VideoCapture(0)
    ds_factor = 0.7

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



        rects = detector.detectMultiScale(gray, scaleFactor=1.05,
	minNeighbors=5, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)
        #rects = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in rects:
            maskOn = True
            nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
            for (xN,yN,wN,hN) in nose_rects:
                if (xN>x and xN<x+w) and (yN>y and yN<y+h):
                    maskOn = False
                # cv2.rectangle(frame, (xN,yN), (xN+wN,yN+hN), (0,255,0), 1)
                break
            
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5)
            for (xM,yM,wM,hM) in mouth_rects:
                if (xM>x and xM<x+w) and (yM>y and yM<y+h):
                    maskOn = False
                # cv2.rectangle(frame, (xM,yM), (xM+wM,yM+hM), (0,255,0), 1)
                break

            if maskOn:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Mask On", (int(x), int(y)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (36,255,12), 2)
                #cv2.imwrite("MaskOn.png", frame)
                break
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Mask Off", (int(x), int(y)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (36,12,255), 2)
                #cv2.imwrite("MaskOff2.png", frame)
                break
        cv2.imshow('Mask Detector', frame)
                

        key = cv2.waitKey(1)
        if key == ord("q"):
            break



if __name__ == "__main__":
    main()