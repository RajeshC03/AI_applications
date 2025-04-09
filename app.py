import cv2
import mediapipe as mp
import pyttsx3
import threading
import time

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Text-to-Speech (TTS) Engine
try:
    tts_engine = pyttsx3.init(driverName="espeak")
except Exception:
    tts_engine = pyttsx3.init()  # Fallback to default engine

# Configure TTS Engine
tts_engine.setProperty("rate", 150)  # Adjust speech speed
tts_engine.setProperty("volume", 1.0)  # Max volume

# Set a default English voice
voices = tts_engine.getProperty("voices")
for voice in voices:
    if "english" in voice.id.lower():
        tts_engine.setProperty("voice", voice.id)
        break

# Function to speak in a separate thread
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()
    time.sleep(0.5)

# Function to recognize hand gestures
def recognize_gesture(hand_landmarks):
    if hand_landmarks:
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        if (index_tip.y < hand_landmarks.landmark[6].y and
            middle_tip.y < hand_landmarks.landmark[10].y and
            ring_tip.y < hand_landmarks.landmark[14].y and
            pinky_tip.y < hand_landmarks.landmark[18].y):
            return "Hi"

        if (index_tip.y > hand_landmarks.landmark[6].y and
            middle_tip.y > hand_landmarks.landmark[10].y and
            ring_tip.y > hand_landmarks.landmark[14].y and
            pinky_tip.y > hand_landmarks.landmark[18].y):
            return "Good Morning"
    
    return "Unknown"

# Start Webcam
cap = cv2.VideoCapture(0)
prev_gesture = ""
last_spoken_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_text = "Waiting..."
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_text = recognize_gesture(hand_landmarks)

            if gesture_text != prev_gesture and gesture_text != "Unknown":
                current_time = time.time()
                if current_time - last_spoken_time > 1.7:
                    prev_gesture = gesture_text
                    last_spoken_time = current_time
                    threading.Thread(target=speak_text, args=(gesture_text,), daemon=True).start()

    cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
