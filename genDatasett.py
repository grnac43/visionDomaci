import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Function to extract hand landmarks from an image
def extract_hand_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()
    return None


klase = ['kamen','papir','makaze']
dataset_path = 'dataset/'
data = []
labels = []

for k in klase:
    cap = cv2.VideoCapture(f'slike/{k}%1d.jpg')

    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is successfully captured
        if not ret:
            break
        landmarks = extract_hand_landmarks(frame)
        if landmarks is not None:
            # Append landmarks to data
            data.append(landmarks)

            # Extract class label from the filename or manually label your data
            # In this example, assuming filenames are like "class_1.jpg", "class_2.jpg", etc.
            if k == 'kamen':
                label=0
            elif k == 'papir':
                label=1
            elif k == 'makaze':
                label=2
            labels.append(label)
        # Display the frame
        cv2.imshow('Frame', frame)


        # Check for key events
        key = cv2.waitKey(1)

    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

data = np.array(data)
labels = np.array(labels)

# Create a DataFrame using pandas
df = pd.DataFrame(data, columns=[f'point_{i}' for i in range(63)])  # Assuming 21 hand landmarks with x, y, and z coordinates
df['label'] = labels

# Save the DataFrame to a CSV file
csv_filename = f'{dataset_path}hand_landmarks_dataset.csv'
df.to_csv(csv_filename, index=False)

print(f'Dataset saved to {csv_filename}')