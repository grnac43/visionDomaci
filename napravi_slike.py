import cv2

# Open video capture (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Variable to track the image index
image_index = 32

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is successfully captured
    if not ret:
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for key events
    key = cv2.waitKey(1)

    # Break the loop when 'Esc' key is pressed
    if key == 27:
        break

    # Save the frame as a JPEG image when 'Space' key is pressed
    elif key == 32:  # 32 is the ASCII code for 'Space' key
        image_index += 1
        image_filename = f'slike/papir{image_index}.jpg'
        cv2.imwrite(image_filename, frame)
        print(f'Frame saved as {image_filename}')

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
