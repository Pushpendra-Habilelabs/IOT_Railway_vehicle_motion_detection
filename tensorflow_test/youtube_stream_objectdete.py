import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd
from imutils.video import VideoStream, FPS
import imutils
import time
import streamlink


# Carregar modelos
detector = hub.load("efficientdet_lite2_detection_1 (1)")

labels = pd.read_csv('labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

#cap = cv2.VideoCapture(0)

#url = 'https://www.youtube.com/watch?v=d_1__1ub2pU'

#streams = streamlink.streams(url)

#cap = cv2.VideoCapture(streams["best"].url)
cap = cv2.VideoCapture('obj test.mp4')
#cap = VideoStream(src='rtsp://habilelabs:qwerty123@182.1.0.102:554/Streaming/channels/401').start()
time.sleep(2.0)
fps = FPS().start()

frame = cap.read()
#frame = imutils.resize(frame, width =1100)

while(True):
    #Capture frame-by-frame
    start = time.time()
    ret, rgb = cap.read()
    rgb = cv2.resize(rgb, (2880, 1500))
    
    #Resize to respect the input_shape
    #rgb = imutils.resize(frame, width= 1100 )

    #Convert img to RGB
    #rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    #Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    #Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    boxes, scores, classes, num_detections = detector(rgb_tensor)
    
    pred_labels = classes.numpy().astype('int')[0]
    
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    end = time.time()

    print("[INFO] it took {:.6f} seconds".format(end - start))
   
   #loop throughout the detections and place a box around it  
    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.1:
            continue

        if label == "car" or label == "motorbike" or label == "truck":   
            #print(label)
            score_txt = str(round(score*100))
            rgb = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),5)      
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb,str(label + "("+score_txt+")"),(xmin, ymin-10), font, 1, (255,0,0), 2, cv2.LINE_AA)
        #Display the resulting frame
        cv2.imshow('black and white',rgb)



    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()