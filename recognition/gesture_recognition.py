import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Define gestures based on landmarks
gestures = {
    "thumbs_up": lambda landmarks: landmarks[4][1] < landmarks[3][1] < landmarks[2][1] and landmarks[4][0] > landmarks[3][0],  # Thumb higher and to the right
    "peace_sign": lambda landmarks: landmarks[8][1] < landmarks[6][1] < landmarks[5][1] and landmarks[12][1] < landmarks[10][1] < landmarks[9][1] and landmarks[16][1] > landmarks[14][1]  # Peace with two fingers up and ring finger down
}

# Variables to track consistent gestures
frame_counter = 0
consistent_gesture = None

# Start webcam feed
cap = cv2.VideoCapture(0)
last_gesture = None  # Track the last spoken gesture to avoid repeating it

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a selfie view and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        result = hands.process(rgb_frame)

        # Draw hand landmarks and check for gestures
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                h, w, _ = frame.shape
                landmarks = np.array([[int(x * w), int(y * h)] for x, y in landmarks])

                # Recognize gestures based on landmarks
                detected_gesture = None
                for gesture, func in gestures.items():
                    if func(landmarks):
                        detected_gesture = gesture
                        break

                # Check if detected gesture is consistent across frames
                if detected_gesture == consistent_gesture:
                    frame_counter += 1
                else:
                    frame_counter = 0
                    consistent_gesture = detected_gesture

                # Only recognize gesture if consistent for several frames
                if frame_counter > 5 and detected_gesture != last_gesture:
                    cv2.putText(frame, detected_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    engine.say(detected_gesture)  # Speak the detected gesture
                    engine.runAndWait()
                    last_gesture = detected_gesture  # Update last spoken gesture

        # Display the result
        cv2.imshow('Gesture Recognition with TTS', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
