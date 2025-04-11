import cv2
import mediapipe as mp
import numpy as np

# Initialize Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Create a Drawing Canvas and Color Selection
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
current_color = (255, 0, 0)  # Default is Blue

# Separate previous points for each hand
prev_coords = {
    'Left': None,
    'Right': None
}

# Function to extract index finger positions
def get_both_index_finger_positions(results, frame_shape):
    finger_positions = {
        'Left': None,
        'Right': None
    }

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_tip.x * frame_shape[1])
            y = int(index_tip.y * frame_shape[0])
            finger_positions[hand_label] = (x, y)

    return finger_positions

# Draw color palette on screen
def draw_palette(frame):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]  # Blue, Green, Red, Black
    positions = [(50, 50), (150, 50), (250, 50), (350, 50)]

    for color, pos in zip(colors, positions):
        cv2.rectangle(frame, pos, (pos[0] + 50, pos[1] + 50), color, -1)

    return colors, positions

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    colors, positions = draw_palette(frame)
    finger_positions = get_both_index_finger_positions(results, frame.shape)

    for hand_label in ['Left', 'Right']:
        pos = finger_positions[hand_label]

        if pos is not None:
            x, y = pos

            # Color selection
            for i, rect in enumerate(positions):
                if rect[0] < x < rect[0] + 50 and rect[1] < y < rect[1] + 50:
                    current_color = colors[i]

            # Drawing (avoid color palette area)
            if y > 100:
                if prev_coords[hand_label] is None:
                    prev_coords[hand_label] = (x, y)

                cv2.line(canvas, prev_coords[hand_label], (x, y), current_color, 15)
                prev_coords[hand_label] = (x, y)
        else:
            prev_coords[hand_label] = None

    # Overlay the drawing on video
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("AI Drawing App - Both Hands", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

