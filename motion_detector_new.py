# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
from darkflow.net.build import TFNet
import argparse
import datetime
import imutils
import time
import cv2
import sys

options = {"model": "tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights", "threshold": 0.6 , "gpu" : 0.4}	
tfnet = TFNet(options)

def main(vid1):
        cap = cv2.VideoCapture(vid1)

        # initialize the first frame in the video stream
        prevFrame = None
        prevObjs = []
        framecount = 0

        # loop over the frames of the video
        while True:
                # grab the current frame and initialize the occupied/unoccupied
                # text
                (grabbed, frame) = cap.read()

                # if the frame could not be grabbed, then we have reached the end
                # of the video
                if not grabbed:
                        break

                #initialize first contour frame if none
                if prevFrame is None:
                        prevFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        prevFrame = cv2.GaussianBlur(prevFrame, (21, 21), 0)

                #get all contours in the current frame. Do this every frame
                contours = findContours(prevFrame, frame)

                #if there are existing objects, update their contours. Do this every frame
                if framecount%5 == 0:
                        if len(prevObjs) > 0:
                                updateObjectContours(prevObjs, contours)


                #actually checking for objects that have left        
                if framecount%25 == 0:
                        print("In frame ", framecount)
                        #get current objects in the frame from DNN. Only run when available
                        currentObjs = getObjects(frame, contours)
                        #if there are old object
                        if len(prevObjs) > 0:
                                checkMatches(prevObjs, currentObjs)

                        prevObjs = currentObjs
                        cv2.imwrite("frame"+str(framecount)+".png", frame)
                        print("\n\n")





                #prevObjs = currentObjs

                #prevFrame = frame
                framecount = framecount + 1

        # cleanup the camera and close any open windows
        cap.release()
        cv2.destroyAllWindows()

#find all contours in a frame
def findContours(prevFrame, frame):
        # resize the frame, convert it to grayscale, and blur it
        #frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if prevFrame is None:
                prevFrame = gray
                return
        
        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(prevFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        contours = []

        # loop over the contours
        for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < 500:
                        continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                #get midpoints of contours in the array
                contour = {'xmid' : x + w/2, 'ymid' : y + h/2, 'xmin' : x, 'ymin' : y, 'xmax' : x+w, 'ymax' : y+h}
                contours.append(contour)
                
        # show the frame and record if the user presses a key
                #cv2.imshow("Security Feed", frame)
        return contours

#get all the frames
def getObjects(frame, contours):
        #fheight, fwidth, _ =
        #fheight, fwidth, _ = frame.shape
	result = tfnet.return_predict(frame)
	objs = []
	fheight, fwidth, _ = frame.shape

	for item in result:
                xmin = item['topleft']['x']
                xmax = item['bottomright']['x']
                ymin = item['topleft']['y']
                ymax = item['bottomright']['y']
                print("ymin and ymax: ", ymin, " and ", ymax)

                p1 = (item['topleft']['x'], item['topleft']['y'])
                p2 = (item['bottomright']['x'], item['bottomright']['y'])
                cv2.rectangle(frame, p1, p2,(255 ,255, 255), 2, 1)


                #assign a contour to each object
                for contour in contours:
                        ctr = None
                        if int(contour['xmid']) >= xmin and int(contour['xmid']) <= xmax and int(contour['ymid']) >= ymin and int(contour['ymid']) <= ymax:
                                #only add the object if an contour is found for it
                                ctr = contour
                                break
                
                obj = {"type": item['label'], "contour": ctr}
                objs.append(obj)

	return objs

#update all the contours each frame
def updateObjectContours(prevObjs, contours):
        for obj in prevObjs:
                oldContour = obj['contour']
                if oldContour != None:
                        newContour = None
                        for contour in contours:
                                if (int(oldContour['xmid']) >= int(contour['xmin']) and int(oldContour['xmid']) <= int(contour['xmax'])) and (int(oldContour['ymid']) >= int(contour['ymin']) and int(oldContour['ymid']) <= int(contour['ymax'])):
                                        newContour = contour
                                        break
                        obj['contour'] = newContour

#check for matches between frames..
def checkMatches(prevObjs, currentObjs):
        filtered = []
        for prevObj in prevObjs:
                for currentObj in currentObjs:
                        if prevObj['type'] != currentObj['type']:
                                continue
                        if prevObj['contour'] == None or currentObj['contour'] == None:
                                continue
                        if int(prevObj['contour']['xmid']) == int(currentObj['contour']['xmid']) and int(prevObj['contour']['ymid']) == int(currentObj['contour']['ymid']):
                                print("match found")
                                filtered.append(prevObj)

        tmp = [i for i in prevObjs if i not in filtered]
        print("Adding ", tmp)

                        

if __name__ == "__main__":
        try:
        	vid1 = sys.argv[1]
        except IndexError:
                print ("You must specify a video")
                sys.exit()

        print("test")
        main(vid1)
