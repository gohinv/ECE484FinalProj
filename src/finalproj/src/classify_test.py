import cv2
import os

def main():

    cascade_path = '../cascade/stop_sign_classifier.xml'

    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found at {cascade_path}")

    # Load the pre-trained Haar Cascade classifier for stop signs
    stop_sign_cascade = cv2.CascadeClassifier(cascade_path)

    # Open the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stop_signs = stop_sign_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # bounding boxes
        for (x, y, w, h) in stop_signs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # display the resulting frame
        cv2.imshow('Stop Sign Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
