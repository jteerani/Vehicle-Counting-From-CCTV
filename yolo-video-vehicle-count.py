# -*- coding: utf-8 -*-
"""

@author: Jinjutha T.
"""
# import the necessary packages
import numpy as np
import imutils
import time
import cv2
from datetime import datetime
from imutils.video import FPS
from imutils.video import FileVideoStream

now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
strat_run = dt_string

from sort import *
tracker = Sort()
memory = {}
counter_car = 0
counter_motocycle = 0

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

LABELS = ["bike","car","truck"]

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")

# derive the paths to the YOLO weights and model configuration
# weightsPath = "Project\yolo-coco\yolov4.weights"
# configPath = "Project\yolo-coco\yolov4.cfg"
weightsPath = "Project\model\custom_yolov4_best.weights"
configPath = "Project\model\custom_yolov4.cfg"


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
input = "Project\input\entrance12.mp4"
vs = cv2.VideoCapture(input)
writer = None
(W, H) = (None, None)

# initialize the video stream, pointer to output video file, and
# frame dimensions
video_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Specifying coordinates for a default line 
x1_line = video_height//4 + video_height
y1_line = 0
x2_line = video_height//4 + video_height
y2_line = video_width
line = [(x1_line, y1_line), (x2_line, y2_line)]

frameIndex = 0
# start the frames per second throughput estimator

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

confidence_set  = 0.7
threshold = 0.3

# loop over frames from the video file stream
while True:

    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    center = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_set:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                center.append(int(centerY))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_set, threshold)
    
    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])
            # print(confidences[i])
            # print(center[i])
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)
    
    boxes = []
    indexIDs = []
    c = []
    
    previous = memory.copy()

    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            
            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                # p0 = (x1_line,y1_line)
                # p1 = (x2_line,y2_line)
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                cv2.line(frame, p0, p1, color, 3)


                if intersect(p0, p1, line[0], line[1]):
                    if(LABELS[classIDs[i]] in ["car","truck"]):
                        counter_car += 1
                        print(LABELS[classIDs[i]],str(counter_car))
                    elif(LABELS[classIDs[i]] in ["bike"]):
                        counter_motocycle += 1
                        print(LABELS[classIDs[i]],str(counter_motocycle))

                        

            text1 = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            # text = "{}".format(indexIDs[i])
            # print("---------------text-------------",text)
            # print("---------------text1-------------",text1)
            cv2.putText(frame, text1, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # cv2.imshow('yolo', frame)
            i += 1

    # draw line
    cv2.line(frame, line[0], line[1], (0, 255, 255), 4)

    # draw counter
    txt = "Car = "+str(counter_car)+ "    " +"Motorcycle = "+str(counter_motocycle)
    txt_car = "Vehicle counting = " + str(counter_car)
    cv2.putText(frame, txt, (50,150), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0) , 2)
    cv2.imshow('Frame', frame)

	# check if the video writer is None
    if writer is None:
		# initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("Project\output\out1.avi", fourcc, 30,(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

	# write the output frame to disk
    writer.write(frame)
    

	# increase frame index
    frameIndex += 1
    # fps.update()

    if frameIndex >= total:
        print("[INFO] cleaning up...")
        writer.release()
        vs.release()
        exit()

    if cv2.waitKey(1) & 0xFF == ord('q'):break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
end_run = dt_string
print("Running timestamp =", strat_run)
print("Running timestamp =", end_run)