import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle

scaler_params = []
with open('scaler_params.pkl', 'rb') as file:
    scaler_params = pickle.load(file)

scaler_m = scaler_params['mean']
scaler_s = scaler_params['scale']
# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Load the trained TensorFlow model
model = load_model('hand_landmarks_classifier_model')

# Map class labels to human-readable gestures
gesture_labels = {
    0: 'Kamen',
    1: 'Papir',
    2: 'Makaze'
    # Add more labels based on your dataset
}

# Open video capture (0 represents the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    # frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hand module
    results = hands.process(rgb_frame)

    # Get hand landmarks if detected
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        hand_data = np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()

        # Standardize the hand landmarks using the same scaler used during training
        hand_data = (hand_data - scaler_m) / scaler_s

        # Reshape the data to match the input shape expected by the model
        hand_data = hand_data.reshape(1, -1)

        # Make a prediction using the trained model
        prediction = model.predict(hand_data)
        predicted_class = np.argmax(prediction)

        # Display the gesture label on the frame
        gesture_label = gesture_labels.get(predicted_class, 'Unknown Gesture')
        cv2.putText(frame, f'Predicted Gesture: {gesture_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Classification', frame)

    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
