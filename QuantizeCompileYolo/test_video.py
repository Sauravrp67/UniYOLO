import cv2
import time

cap = cv2.VideoCapture("/home/saurav/Desktop/Internship/ML-Internship-Saurav-Paudel/Paper_Implementation/ObjectDetection/UniYOLO/QuantizeCompileYolo/V3/test_videos/videoplayback.webm")
original_fps = cap.get(cv2.CAP_PROP_FPS)
print(original_fps)
delay_ms = int(1000/original_fps)
print(delay_ms)
while True:
    t0 = time.time()
    ret,frame = cap.read()
    if not ret:
        break
    fps = (1/(time.time() - t0)) 
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video",frame)
    if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()