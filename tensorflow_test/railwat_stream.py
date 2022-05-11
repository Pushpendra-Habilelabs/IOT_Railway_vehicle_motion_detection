import cv2
import numpy as np
import time
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import serial


"""
cap = cv2.VideoCapture('rtsp://habilelabs:qwerty123@182.1.0.102:554/Streaming/Channels/301')
#'rtsp://habilelabs:qwerty123@182.1.0.102:554/Streaming/Channels/301'


# kernel for image dilation
kernel = np.ones((4,4),np.uint8)

ret, thresh = cv2.threshold(cap, 30, 255, cv2.THRESH_BINARY)

while(True):
    # image thresholding
    ret, test = cap.read()
    grayA = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayA, 100, 255, cv2.THRESH_BINARY)
    
    
    # image dilation
    dilated = cv2.dilate(thresh,kernel,iterations = 1)

   
    # find contours
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # shortlist contours appearing in the detection zone
    valid_cntrs = []
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        if (x <= 1400) & (y >= 280) & (y <= 500)  & (cv2.contourArea(cntr) >= 150):
            cv2.rectangle(test,(x,y),(x+w,y+h),(0,255,0),2)
            if (y >= 280) & (y <= 500) & (cv2.contourArea(cntr) < 50):
                break 
            valid_cntrs.append(cntr)
    
    # add contours to original frames
    #cv2.drawContours(test, valid_cntrs, -1, (0,0,255), 2)
    cv2.line(test, (1400, 280), (0, 280),(255,0,0), 3)
    cv2.line(test, (1400, 580), (0, 580),(255,0,0), 3)
    cv2.imshow('frame', test)
    
    #cv2.imshow('frame',thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


"""
#vs = VideoStream(src='rtsp://habilelabs:qwerty123@182.1.0.102:554/Streaming/Channels/301').start()

vs = cv2.VideoCapture('obj test.mp4')
#vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

#ser = serial.Serial("/dev/ttyUSB0")

frame = vs.read()
#frame = imutils.resize(frame, width = 1100)

yolo = "yolo/"

labelsPath = yolo+"coco.names"
weightsPath = yolo+"yolov3.weights"
configPath = yolo+"yolov3.cfg"

LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")



print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
print(net)

# set CUDA as the preferable backend and target
print("[INFO] setting preferable backend and target to CUDA...")
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getUnconnectedOutLayersNames()
#print(ln)

#ser.setRTS(1) # Signal Out to FTDI232 RTS Pin NO Signal

obj = 0

while True:
            
    # grab the frame from the threaded video stream and resize it

    
    # to have a maximum width of 400 pixels
    ret, frame = vs.read()
    #frame = imutils.resize(frame, width=1100)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    #cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5

    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []


    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.001:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.1,
        0.3)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            #cv2.circle(image, (CX,CY), radius=5, color=(0, 0, 255), thickness=-1)
            
            if (x >= 50) & (x <= 1050) :
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
                obj = 'object'
                if (x < 50) & (x > 1050) or ((x+w) > 1050):
                    obj = 'null'
                    break 
        

    else:
        obj = 'null'
        print(len(idxs))
        
                
    cv2.line(frame, (200, 800), (200, 0),(255,0,0), 3)
    cv2.line(frame, (900, 800), (900, 0),(255,0,0), 3)
    if obj == 'object':                
        #ser.setRTS(0) # Signal Out to FTDI232 RTS Pin Signal
        print("object")
        time.sleep(2)

    else:
        #ser.setRTS(1) # Signal Out to FTDI232 RTS Pin NO Signal
        print("Null")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


    


"""
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))







"""
