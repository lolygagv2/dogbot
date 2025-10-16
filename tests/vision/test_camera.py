import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

print("✅ Camera opened. Press Ctrl+C to stop.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed.")
        break
    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()