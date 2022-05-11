import streamlink
import cv2

url = 'https://www.youtube.com/watch?v=VIwzSSGbxZw'

streams = streamlink.streams(url)
#print(streams)
#cap = cv2.VideoCapture(streams["720p"].url)
cap = cv2.VideoCapture(streams["best"].url)

while True:
    ret, frame = cap.read()
    #print(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()