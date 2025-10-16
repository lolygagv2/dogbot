import cv2
import mediapipe as mp

pose = mp.solutions.pose.Pose()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
        print("✅ Pose detected.")

    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()