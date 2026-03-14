import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    if ret:
        print("Working camera index:", i)
        cv2.imshow("Test", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    cap.release()