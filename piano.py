import cv2
import mediapipe as mp
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import time
import threading

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load piano sound files (Ensure you have 10 different note files: 'note1.mp3', ..., 'note10.mp3')
note_files = [f'notes/note{i}.mp3' for i in range(1, 11)]
notes = [AudioSegment.from_mp3(f) for f in note_files]

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increase height

# Map fingers to notes for left and right hand
left_finger_tips = [4, 8, 12, 16, 20]  # Left Thumb, Index, Middle, Ring, Pinky
right_finger_tips = [4, 8, 12, 16, 20]  # Right Thumb, Index, Middle, Ring, Pinky
finger_joints = [3, 6, 10, 14, 18]  # Corresponding lower joints for comparison
note_cooldowns = {}  # Track last played time
cooldown_time = 0.5  # 0.5 seconds cooldown
finger_states = {}  # Track whether a finger is currently raised
finger_states_prev = {}  # Track previous state of each finger

# Function to play note
def play_note(note_index):
    play(notes[note_index])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    current_time = time.time()
    
    if result.multi_hand_landmarks:
        hand_count = 0  # Track left & right hands separately
        for hand_landmarks in result.multi_hand_landmarks:
            if hand_count == 0:
                finger_tips = left_finger_tips
                base_index = 0  # Map to note1 - note5
            else:
                finger_tips = right_finger_tips
                base_index = 5  # Map to note6 - note10
            
            for i, (tip_idx, joint_idx) in enumerate(zip(finger_tips, finger_joints)):
                tip = hand_landmarks.landmark[tip_idx]
                joint = hand_landmarks.landmark[joint_idx]
                
                note_index = base_index + i  # Assign correct note
                is_raised = tip.y < joint.y  # Finger is raised
                previously_raised = finger_states.get(note_index, False)
                
                # Play sound only when the finger transitions from raised to lowered
                if previously_raised and not is_raised and current_time - note_cooldowns.get(note_index, 0) > cooldown_time:
                    note_cooldowns[note_index] = current_time
                    threading.Thread(target=play_note, args=(note_index,), daemon=True).start()
                
                # Update finger states
                finger_states[note_index] = is_raised
                
                h, w, _ = frame.shape
                cx, cy = int(tip.x * w), int(tip.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            
            hand_count += 1  # Move to right hand
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Air Piano", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

