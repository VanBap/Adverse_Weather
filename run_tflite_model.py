from imutils.video import VideoStream
from imutils.video import FPS
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
import tensorflow as tf
import argparse
import imutils
import time
import cv2
import draw_label

if __name__ == "__main__":
    # Load model
    print("[INFO] loading model...")
    interpreter = tf.lite.Interpreter(model_path='F:/CODE_PYCHARM/KhoaLuan/saved_model/ResNet50_Weather_epoch20.tflite')
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    print("input details", input_details)
    print("output details", output_details)


    # Labels array
    CATEGORIES=['CLOUDY','FOG','RAINY','SANDY','SHINE','SNOWY', 'SUNRISE' ]

    # Open the device at the ID 0
    # Use the camera ID based on
    # /dev/videoID needed
    cap = cv2.VideoCapture(0)

    # Check if camera was opened correctly
    if not (cap.isOpened()):
        print("Could not open video device")

    # Fetch one frame at a time from your camera
    while (True):
        # Frame is a numpy array, that you can predict on
        ret, frame = cap.read()

        # Resize frame to (224, 224)
        resized_frame = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(resized_frame, axis=0)

        # Preprocess input data if needed
        input_data = preprocess_input(input_data)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Perform inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process output data as needed
        predicted_weather = CATEGORIES[np.argmax(output_data)]

        # Adding the label on frame
        draw_label.__draw_label(frame, 'Label: {}'.format(predicted_weather), (30, 30), (255, 255, 0))


        # Display the resulting frame
        cv2.imshow("preview", frame)

        # Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
