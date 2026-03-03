import cv2
import numpy as np

def capture_image_from_camera():
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    while True:
        ret, frame = cap.read()  
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Live Camera - Press "s" to capture', frame) 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): 
            captured_frame = frame
            break
        elif key == ord('q'):  
            print("Exiting without capturing.")
            return None

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame


