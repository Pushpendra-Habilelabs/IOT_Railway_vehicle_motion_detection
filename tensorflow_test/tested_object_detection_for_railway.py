from sympy import cxxcode
import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd
from imutils.video import VideoStream, FPS
import imutils
import time
import coremltools as ct


tf.config.set_visible_devices([], 'GPU')
# Carregar modelos
detector = hub.load("efficientdet_lite2_detection_1 (1)")
print(detector)

#detector = ct.convert(detector)

labels = pd.read_csv('labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']
multi_car_direction = []
#cap = cv2.VideoCapture(0)

#url = 'https://www.youtube.com/watch?v=d_1__1ub2pU'

#streams = streamlink.streams(url)

#cap = cv2.VideoCapture(streams["best"].url)

cap = cv2.VideoCapture('obj test.mp4')
#cap = VideoStream(src='rtsp://habilelabs:qwerty123@182.1.0.102:554/Streaming/channels/401').start()
#time.sleep(2.0)
fps = FPS().start()

gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
frame = cap.read()
#frame = imutils.resize(frame, width =1100)

while(True):

    car_direction = []
    #Capture frame-by-frame
    start = time.time()
    
    ret, rgb = cap.read()
    print(rgb)
    break
   
    #rgb = cv2.resize(rgb, (2880, 1500))
    #Resize to respect the input_shape
    #rgb = imutils.resize(frame, width= 1100 )

    #Convert img to RGB
    #rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    #Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    #Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    """
    boxes, scores, classes, num_detections = detector(rgb_tensor)
    pred_labels = classes.numpy().astype('int')[0]
    
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]
    cv2.line(rgb, (1740, 0), (0, 750),(255,0,0), 3)
    cv2.line(rgb, (2000, 1500), (2580, 0),(255,0,0), 3)
   #loop throughout the detections and place a box around it  

    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.15:
            continue
        if label == "car" or label == "motorbike" or label == "truck":   
            if (xmax >= 900) & (xmin <= 2000):
                #print(label)
                cx = int(((xmax - xmin)/2) + xmin)
                cy = int(((ymax - ymin)/2) + ymin)
                car_direction.append(cx)
                score_txt = str(round(score*100))
                rgb = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),5)      
                rgb = cv2.circle(rgb, (cx, cy), radius= 10, color=(0, 165, 255), thickness=-1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(rgb,str(label + "("+score_txt+")"),(xmin, ymin-10), font, 1, (255,128,0), 2, cv2.LINE_AA)
                cx_low = cx-7
                cx_high = cx+7
                cx_change_low = cx-20
                cx_change_high = cx+20
                list_i = []
                for i in range (cx_low, cx_high, 1):
                    list_i.append(i)
                    
                multi_list_i = []
                for i in range (cx_change_low, cx_change_high, 1):
                    multi_list_i.append(i)
                    
                final_obj_list = [x for x in multi_list_i if x not in list_i]

                for i in range(cx_change_low, cx_change_high, 1):
                    if (i in multi_car_direction) and (i in list_i) and (i not in final_obj_list): 
                        cv2.putText(rgb,"Stop",(xmax, ymin-10), font, 1, (0,255,255), 2, cv2.LINE_AA)
                        break

                    elif (i in multi_car_direction) and (i in final_obj_list)and (i not in list_i):
                        cv2.putText(rgb,"Motion",(xmax, ymin-10), font, 1, (0,255,255), 2, cv2.LINE_AA)
                        break
                        

            
                        

                #print(label)
            elif (xmax < 900) & (xmin > 2000):
                break 

    

    multi_car_direction = ([y for y in multi_car_direction if y not in multi_list_i])

    multi_car_direction.extend(car_direction)  

      
    #Display the resulting frame
    """
    end = time.time()
    print("[INFO] it took {:.6f} seconds".format(end - start))
   
    cv2.imshow('black and white',rgb)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()