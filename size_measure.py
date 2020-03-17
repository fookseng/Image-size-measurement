        # import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import json

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
    help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# #cvInRangeS(imgHSV, cvScalar(20, 100, 100), cvScalar(30, 255, 255), imgThreshed) for yellow object.
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower = np.array([20,100,100])
# upper = np.array([30,255,255])
# ####### mask1 = mask1+mask2
# mask_hsv = cv2.inRange(hsv, lower, upper)
# output_hsv = cv2.bitwise_and(image, image, mask = mask_hsv)
#
# #cv2.imshow("mask_hsv",output_hsv)
# #key = cv2.waitKey(0)
# cv2.imshow("yellow",output_hsv)

gray=image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("grayyy",gray)
for l in range(0, gray.shape[0]):
    for m in range(0, gray.shape[1]):
        pixel = gray[l, m]
        if pixel <= 100:
            gray[l, m] = 0
        elif pixel == 0:
           gray[l, m] = 255
        else:
            gray[l, m] = 255
width = gray.shape[1]
height = gray.shape[0]
print(width,height)
#cv2.imshow("gray",gray)
# gray = cv2.cvtColor(output_hsv, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (7, 7), 0)
#
# edged = cv2.Canny(image, 120, 150)
# edged = cv2.dilate(edged, None, iterations=1)
# edged = cv2.erode(edged, None, iterations=1)

#cv2.imshow("grayyyy",edged)

#key = cv2.waitKey(0)

# find contours in the edge map
cnts = cv2.findContours(gray.copy(), cv2.RETR_TREE,
    cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts,"top-to-bottom")
pixelsPerMetric = None
dic = {"data": pixelsPerMetric}


# loop over the contours individually
for c in cnts:
    square = False
    approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
    if len(approx) == 4 :
        square = True
    if square == True and cv2.contourArea(c)>1:
        print("is square")
        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        #if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]
        dic["data"] = pixelsPerMetric
        print("pixel per metric is:" + str(pixelsPerMetric))
        print("width: "+ str(width/pixelsPerMetric) + "mm")
        print("height: "+str(height/pixelsPerMetric) + "mm")
        #np.save("ppm",pixelsPerMetric)
        #data = open('data.txt', 'x')
        #with open('data.txt','w') as f:
        #    f.write(str(pixelsPerMetric))
        with open('data.json', 'w') as outfile:
            json.dump(dic, outfile)
        ##cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Image', 600, 600)
        cv2.imshow("Image", orig)
        cv2.waitKey(0)
    else:
        cv2.waitKey(0)
