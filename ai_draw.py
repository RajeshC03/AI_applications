# Import Required Libraries
import cv2
import mediapipe as mp
import numpy as np

# Initialize Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Create a Drawing Canvas and Color Selection
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Wider Black Canvas
current_color = (255, 0, 0)  # Default color is Blue

# Initialize previous coordinates for smooth drawing
prev_x, prev_y = None, None  

# Function to Get Finger Position
def get_finger_position(results, frame_shape):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * frame_shape[1]), int(index_finger_tip.y * frame_shape[0])
            return x, y
    return None, None

# Create a Color Palette Function
def draw_palette(frame):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]  # Blue, Green, Red, Black
    positions = [(50, 50), (150, 50), (250, 50), (350, 50)]  # Button positions

    for i, (color, pos) in enumerate(zip(colors, positions)):
        cv2.rectangle(frame, pos, (pos[0] + 50, pos[1] + 50), color, -1)
    return colors, positions

# Process Webcam Feed & Implement Smooth Drawing
cap = cv2.VideoCapture(0)  # Open webcam
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for better interaction
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    colors, positions = draw_palette(frame)  # Draw color selection buttons

    x, y = get_finger_position(results, frame.shape)

    if x is not None and y is not None:
        # Check if user selects a color
        for i, pos in enumerate(positions):
            if pos[0] < x < pos[0] + 50 and pos[1] < y < pos[1] + 50:
                current_color = colors[i]

        # Smooth Drawing (Avoid the color palette area)
        if y > 100:
            if prev_x is None or prev_y is None:
                prev_x, prev_y = x, y  # First touch point

            # Draw a line instead of dots for smooth strokes
            cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, 15)  # Thicker brush

            prev_x, prev_y = x, y  # Update previous coordinates
    else:
        prev_x, prev_y = None, None  # Reset when finger is lifted

    # Merge drawing canvas with the video feed
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("AI Drawing App", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

