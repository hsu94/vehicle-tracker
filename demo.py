from darkflow.net.build import TFNet
import cv2
import os
import glob
import shutil
import copy
import numpy as np
import sys

#let these be user configurations?
direction = 1 #assuming 0 = left, 1 = right
fps = 60 #number of frames taken per second
length = 50 #distance of the entire length of the frame horizontally
speed = 5 #average speed of vehicle horizontally in m/s

avgNumFrames = length/speed * fps; #average number of frames a vehicle will be in


options = {"model": "tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights", "threshold": 0.6 , "gpu" : 0.4}	
tfnet = TFNet(options)

def main(vid1):
        cap = cv2.VideoCapture(vid1)

        framecount = 0
        existingObjs = []
        newObjs = []

        while(1):
                ret, frame = cap.read()
                #if valid frame
                if ret == True:
                        #get frame width, height
                        fheight, fwidth, _ = frame.shape

                        newObjs = []
                        #only do this when the thread is ready

                        for obj in existingObjs:
                                #update the trackers each frame
                                ok, bbox = obj['tracker'].update(frame)
                                if ok:
                                        obj['box'] = bbox
                        
                        if framecount % 25 == 0:
                                newObjs = getObjects(frame, framecount)
                                print("Checking in Frame " , framecount)
                                for index,obj in enumerate(existingObjs):
                                        if findMatch(obj['type'], obj['box'], newObjs, obj['frame']) == False:
                                                                        
                                                print(obj['type'] , " ADDED IN FRAME " , framecount)
                                        p1 = (int(obj['box'][0]), int(obj['box'][1]))
                                        p2 = (int(obj['box'][0] + obj['box'][2]), int(obj['box'][1] + obj['box'][3]))
              
                                        cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
                                        
                                for newObj in newObjs:                                                  
                                    p3 = (int(newObj['box'][0]), int(newObj['box'][1]))
                                    p4 = (int(newObj['box'][0] + newObj['box'][2]), int(newObj['box'][1] + newObj['box'][3]))
                                    cv2.rectangle(frame, p3, p4, (255 ,255, 255), 2, 1)

                                cv2.imshow('frame', frame)
                                #cv2.imwrite("frame"+str(framecount)+".png", frame)
                                print("\n")


                                existingObjs = newObjs
			


                        framecount += 1
                        #if framecount > 200:
                         #       break


                else:
                	break


        cap.release()
        cv2.destroyAllWindows()



def getObjects(frame, framecount):
        #fheight, fwidth, _ =
        #fheight, fwidth, _ = frame.shape
	result = tfnet.return_predict(frame)
	objs = []
	fheight, fwidth, _ = frame.shape

	for item in result: 
                x_mid = (item['bottomright']['x'] - item['topleft']['x'])/2
                y_mid = (item['topleft']['y'] - item['bottomright']['y'])/2

                tracker = cv2.TrackerKCF_create()
                box = ( item['topleft']['x'], item['topleft']['y'], item['bottomright']['x'] - item['topleft']['x'], item['bottomright']['y'] - item['topleft']['y'])
                ok = tracker.init(frame, box)

                if ok:
                        #print("border is ", fwidth, " box is on  " , (int(box[0]) + int(box[2])))
                        if int(box[0]) <= 5 or (int(box[0]) + int(box[2]) >= fwidth-5) or int(box[1]) <= 5 or (int(box[1]) + int(box[3]) >= fheight-5):
                                print("Not adding darkflow detected object as it is on border")
                        else:
                                obj = {"type": item['label'], "x-mid": x_mid, "y-mid" :y_mid, "frame" : framecount, "ttl": 3, "box" : box, "tracker" : tracker}
                                objs.append(obj)

	return objs

#try to match an object (obj) from the previous frame to any object in the current frame (currentObjs) by checking if they have the 
#same classification type and if their midpoints are close enough together (determined arbitrarily)
#return 0 if no match or 1 if there is a match
def checkMatch(obj, currentObjs, fheight, fwidth):
        diffPerFrame = fwidth/avgNumFrames #difference between same objects in differing frames
        distanceBuffer = diffPerFrame/2
        print("checking match for object at " + str(obj['x-mid']))
        if len(currentObjs) == 0:
        	return None

        for index, currentObj in enumerate(currentObjs):
                frameDiff = currentObj['frame'] - obj['frame']
                print("Expected location: " , obj['x-mid'] + frameDiff*(diffPerFrame))
                print("current object exists at ", currentObj['x-mid'])

                if obj['type'] != currentObj['type']:
                	continue
                if (currentObj['x-mid'] >= obj['x-mid'] + frameDiff*(diffPerFrame - distanceBuffer)) and (currentObj['x-mid'] <= obj['x-mid'] + frameDiff*(diffPerFrame + distanceBuffer)):
                        del(currentObjs[index])
                        return currentObj

                continue

        return None

def findMatch(objType, bbox, newObjs, frame):
        #where the object is expected to be
        xmid = int(bbox[0]) + (int(bbox[2])/2)
        ymid = int(bbox[1]) + (int(bbox[3])/2)

        print("checking for matches with objects from frame ", frame)
        for obj in newObjs:
                xmin = int(obj['box'][0]) 
                xmax = int(obj['box'][0]) + int(obj['box'][2])
                ymin = int(obj['box'][1])
                ymax = int(obj['box'][1]) + int(obj['box'][3])

                print("checking if x-mid: " , xmid , " is between " , xmin , " and " , xmax)
                print("checking if y-mid: " , ymid , " between " , ymin , " and " , ymax)

                if obj['type'] != objType:
                        continue
                
                if (xmid >= xmin and xmid <= xmax and ymid >= ymin and ymid <= ymax):
                        return True

                continue

        return False
                


if __name__ == "__main__":
        try:
        	vid1 = sys.argv[1]
        except IndexError:
                print ("You must specify a video")
                sys.exit()

        print("test")
        main(vid1)


##                                if ok:
##                                        p1 = (int(obj['box'][0]), int(obj['box'][1]))
##                                        p2 = (int(obj['box'][0] + obj['box'][2]), int(obj['box'][1] + obj['box'][3]))
##                                        cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)     

##                        for obj in newObjs:
##                                p1 = (int(obj['box'][0]), int(obj['box'][1]))
##                                p2 = (int(obj['box'][0] + obj['box'][2]), int(obj['box'][1] + obj['box'][3]))
##                                cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
                                
