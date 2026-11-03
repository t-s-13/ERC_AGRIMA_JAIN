import time
import cv2
import pygame
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

pygame.mixer.init()

MODEL_PATH = "hand_landmarker.task"

previous_count = 0

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

# --- HELPER FUNCTION FOR DRAWING SKELETON -------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm
]

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    
    pixel_landmarks = []
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        pixel_landmarks.append((x, y))

    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(frame, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)

    for pt in pixel_landmarks:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
# -----------------------------------------------------------------

sounds_mapping = {
    1: "sounds_swaras/1_Sa.mp3",
    2: "sounds_swaras/2_Re.mp3",
    3: "sounds_swaras/3_Ga.mp3",
    4: "sounds_swaras/4_Ma.mp3",
    5: "sounds_swaras/5_Pa.mp3",
    6: "sounds_swaras/6_Dha.mp3",
    7: "sounds_swaras/7_Ni.mp3",
    8: "sounds_swaras/8_Sa_upper.mp3",
}

swara_names = {
    1: "Sa", 2: "Re", 3: "Ga", 4: "Ma",
    5: "Pa", 6: "Dha", 7: "Ni", 8: "Sa (upper)",
}

cap = cv2.VideoCapture(0)

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    mirrored_frame = cv2.flip(frame, 1) #mirrors the image horizontally (flipCode = 1 means flip around y-axis)

    timestamp_ms = int((time.time() - start_time) * 1000)

    rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    for hand_landmarks in result.hand_landmarks:
        draw_landmarks(mirrored_frame, hand_landmarks)

    total_count = 0

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # for thumb (tip=4, joint=3)
            if hand_landmarks[5].x < hand_landmarks[17].x:
                if hand_landmarks[4].x < hand_landmarks[3].x:
                    total_count+=1
            else:
                if hand_landmarks[4].x > hand_landmarks[3].x:
                    total_count+=1

            # index finger (tip=8, joint=6)
            if hand_landmarks[8].y < hand_landmarks[6].y:
                total_count += 1
            
            # middle finger (tip=12, joint=10)
            if hand_landmarks[12].y < hand_landmarks[10].y:
                total_count += 1
                
            # ring finger (tip=16, joint=14)
            if hand_landmarks[16].y < hand_landmarks[14].y:
                total_count += 1
                
            # pinky finger (tip=20, joint=18)
            if hand_landmarks[20].y < hand_landmarks[18].y:
                total_count += 1

    if total_count != previous_count:
        if total_count in sounds_mapping:
            pygame.mixer.music.load(sounds_mapping[total_count])
            pygame.mixer.music.play()
        previous_count = total_count

    current_swara = swara_names.get(total_count, "")

    cv2.putText(
        mirrored_frame, 
        f"Count: {total_count} ({current_swara})", 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        2.2, 
        (0, 255, 255), 
        5
    )
    # -------------------------------------

    # Checkpoint: Print the number of hands detected (0, 1, or 2)[cite: 5]
    print(len(result.hand_landmarks))
    # --------------------------------------------------------------------------------------

    cv2.imshow('Webcam Feed', mirrored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty('Webcam Feed', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
