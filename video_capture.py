import cv2
import time
cap = cv2.VideoCapture(0)

assert cap.isOpened(), 'Cannot capture source'

frames = 0
start = time.time()

while cap.isOpened():
    ret,frame = cap.read()
    if ret:
        sized = cv2.resize(frame,(608,608))
        # sized = cv2.cvtColor(sized,cv2.COLOR_BGR2RGB)

        cv2.imshow("frame", sized)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        frames += 1
        print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
    else:
        break