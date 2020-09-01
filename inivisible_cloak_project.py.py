import cv2
import time
import numpy as np
#Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#here webcam is reading
#give camera few seconds to capture background with diffrent frames
cap = cv2.VideoCapture(0)

#Allowing the system to sleep for 2 seconds before the webcam starts
time.sleep(2)
count = 0
back = 0

#Capturing the background in different frames
for i in range(60):
    ret, back = cap.read()
back = np.flip(back, axis=1)

# Reading every frame from the webcam i.e.background
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    img = np.flip(img, axis=1)

    # Converting the color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #need to generate the masks to detect red color
    #here these [ ,  , ] are the color value(which need to be removed)
	#  which can be changed according to the cloth which want to use. 
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2

    # Opening and Dilating the mask image
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Created an inverted mask to segment out the red color from the frame
    mask2 = cv2.bitwise_not(mask1)

    # Segmenting the red color part out of the frame using bitwise and with the inverted mask
    res1 = cv2.bitwise_and(img, img, mask=mask2)

    # Create image showing static background frame pixels only for the masked region
    res2 = cv2.bitwise_and(back, back, mask=mask1)

    # Generating the final output and writing
    finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
    out.write(finalOutput)
    cv2.imshow("magic", finalOutput)
    cv2.waitKey(1)


cap.release()
out.release()
cv2.destroyAllWindows()

