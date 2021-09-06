import cv2
import os
import time
import mediapipe as mp
import HandTrackingModule as htm

wCam,hCam=640,480
cap =cv2.VideoCapture(0)

cap.set(3,wCam)
cap.set(4,hCam)
imageList=[]
#address="https://192.168.1.12:4747/video"

folderPath="FingerImage"
mylist = os.listdir(folderPath)
#cap.open(address)

for imPath in mylist:
    image=cv2.imread(f'{folderPath}/{imPath}')
    print(f'{folderPath}/{imPath}')
    imageList.append(image)

print(len(imageList))
pTime=0

detector =htm.handDetector(detectionCon=0.75)

tipIds=[4,8,12,16,20]


while True:
    success ,img =cap.read()

    img=detector.findHands(img)

    lmList=detector.findPosition(img,draw=False)

    if len(lmList) != 0:
        fingers=[]
        #This is for Thumd to check open or not if point 4 of thumb x position is less than piont 3 than our thumds is close
        # If greater than thumb is open
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #This is for all 4 fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]: #Here we are checking the our hand open or close there for we use index finger point 8
                #y position less than 6 index finger than our hand is open or visa versa
                fingers.append(1)
                #print("Index finger open")
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers=fingers.count(1)
        #print(totalFingers)

        h,w,c=imageList[totalFingers-1].shape #It will give the shape of images
        img[0:h,0:w]=imageList[totalFingers-1]  #This is use for to show the image of video capture img[x:height,y:width] manual placing

        #cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,7,(0,0,255),20)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f'FPS {int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image",img)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

