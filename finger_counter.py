import cv2
import time
import os
import re
import HandTrackingModule as htm

# Try to import winsound for Windows beep, otherwise use print('\a')
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("winsound not available, using console beep.")

# Camera settings
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
# Replace with your phone's IP address (shown in IP Webcam app)
#MOBILE_CAM_URL = "http://10.60.0.182:8080/video"  # Use YOUR phone's IP
#cap = cv2.VideoCapture(MOBILE_CAM_URL)
#cap.set(3, wCam)
#cap.set(4, hCam)

# Load finger images and map them to the correct count
folderPath = "FingerImages"
image_files = os.listdir(folderPath)
count_to_img = {}  # dictionary: count -> image

for f in image_files:
    numbers = re.findall(r'\d+', f)
    if numbers:
        num = int(numbers[0])
        # Map 6 to count 0, numbers 1‑5 to counts 1‑5
        if num == 6:
            count = 0
        elif 1 <= num <= 5:
            count = num
        else:
            continue
        img = cv2.imread(os.path.join(folderPath, f))
        if img is not None:
            count_to_img[count] = img

# Create a list where index = finger count, value = image (or None if missing)
overlay_by_count = [None] * 6
for cnt in range(6):
    if cnt in count_to_img:
        overlay_by_count[cnt] = count_to_img[cnt]
    else:
        print(f"Warning: No image for finger count {cnt}")

pTime = 0
detector = htm.handDetector(detectionCon=0.75)

# Landmark indices for finger tips (MediaPipe hand landmarks)
tipIds = [4, 8, 12, 16, 20]

# Sequence state
expected_count = 0          # next expected finger count
last_detected_count = None  # last finger count we acted upon
warning = False
warning_start_time = 0
WARNING_DURATION = 2        # seconds

# Function to play a warning sound
def play_sound():
    if SOUND_AVAILABLE:
        winsound.Beep(1000, 500)  # frequency 1000 Hz, duration 500 ms
    else:
        print('\a')  # ASCII bell (may work on some terminals)

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Camera not working")
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    hand_detected = len(lmList) != 0
    totalFingers = 0

    if hand_detected:
        # Determine hand orientation (left/right) from MediaPipe results
        handedness = "Right"  # default
        if detector.results and detector.results.multi_handedness:
            # We are using handNo=0 (first hand) – adjust if you have two hands
            handedness_label = detector.results.multi_handedness[0].classification[0].label
            handedness = handedness_label  # "Left" or "Right"

        # Calculate finger count with improved thumb logic
        fingers = []

        # Thumb: for right hand, thumb tip x > thumb IP x; for left hand, opposite
        if handedness == "Right":
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Other four fingers (same for both hands)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        # --- Sequence logic ---
        # Only consider a new gesture if the finger count changed
        if last_detected_count is None or totalFingers != last_detected_count:
            if totalFingers == expected_count:
                # Correct step: advance sequence
                expected_count = (expected_count + 1) % 6
                warning = False
            else:
                # Wrong gesture: reset, show warning, and play sound
                expected_count = 0
                warning = True
                warning_start_time = time.time()
                play_sound()
            # Remember this gesture to avoid re‑checking it
            last_detected_count = totalFingers

        # --- Display overlay for the current finger count ---
        if 0 <= totalFingers <= 5 and overlay_by_count[totalFingers] is not None:
            overlay_img = overlay_by_count[totalFingers].copy()
            overlay_img = cv2.resize(overlay_img, (200, 200))
            h, w, _ = overlay_img.shape
            img[0:h, 0:w] = overlay_img

        # Draw rectangle and big number for the finger count
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # --- Show warning if active ---
    if warning:
        elapsed = time.time() - warning_start_time
        if elapsed < WARNING_DURATION:
            cv2.putText(img, "WRONG SEQUENCE! RESET TO 0", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            warning = False

    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()