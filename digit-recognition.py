import cv2
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Dense

'''
    Method used to rescale the frame window,
    params are expected in percentage
'''


def rescale_frame(frame, wpercent=70, hpercent=70):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def create_model():
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_initializer='he_uniform',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model


'''
    Method is used to resize the image to dimension (28, 28) and then reshape
    it to dimension (28,28,1)
'''


def image_conversion(test):
    test = cv2.resize(test, (28, 28))
    test = test.reshape((28, 28, 1))
    test = np.array([test])
    return test


'''
    Method is used to load pre-trained handwritten digit model for prediction
    from the specified path 
'''


def load_keras_model(path):
    model = create_model()
    model.load_weights(path)
    return model


'''
    Method use to write text in frame,
    used for writing predicted value in the frame
'''


def write_on_frame(frame, text):
    cv2.putText(frame, "Prediction is : " + str(text), (20, 300), cv2.FONT_HERSHEY_COMPLEX, 0.7, 255, 2)


'''
    Toggle writing and prediction mode when either of (M, N) is pressed,
    and clear the drawing form the frame by clearing the points deque
'''


def toggle_mode_and_clear_points(points, prediction_mode, writing_mode):
    points.clear()
    writing_mode = not writing_mode
    prediction_mode = not prediction_mode
    return prediction_mode, writing_mode


'''
    Find the maximum contour from the contours array and 
    then draw the circle around the contour and draw a point 
    at its center 
'''


def draw_contours_in_frame(frame, contours):
    c = max(contours, key=cv2.contourArea)

    moment = cv2.moments(c)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))

    if radius > 0.5:
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    return center


def main():
    writing_mode = False
    prediction_mode = False

    # Blue Color Object Range
    blue_low = np.array([110, 100, 100])
    blue_high = np.array([130, 255, 255])

    # Red Color Object Range
    red_low = np.array([0, 0, 255])
    red_high = np.array([0, 0, 255])

    window_masked = "Masked Window Feed"
    window_original = "Original Live Feed"

    points = deque(maxlen=480)
    capture = cv2.VideoCapture(0)
    model = load_keras_model("pre-trained-model/handwritten_digit_model.h5")

    if capture.isOpened():
        flag, frame = capture.read()
    else:
        flag = False

    while flag:

        pressed_key = cv2.waitKey(1)
        flag, frame = capture.read()
        frame = rescale_frame(frame)

        frame = cv2.flip(frame, 1)

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        frame_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(frame_hsv, blue_low, blue_high)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        image, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            center = draw_contours_in_frame(frame, contours)

            if writing_mode:
                points.append(center)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 20)

        red_mask = cv2.inRange(frame, red_low, red_high)
        red_mask = cv2.dilate(red_mask, None, iterations=1)
        red_mask = cv2.erode(red_mask, None, iterations=1)

        cv2.imshow(window_masked, red_mask)

        if prediction_mode:
            prediction = model.predict_classes(test_image)
            write_on_frame(frame, text=prediction[0])

        cv2.imshow(window_original, frame)

        if pressed_key & 0xFF == ord("n") or pressed_key & 0xFF == ord("n"):
            test_image = image_conversion(red_mask)
            points.clear()
            writing_mode = not writing_mode
            prediction_mode = not prediction_mode
        elif pressed_key & 0xFF == ord('m') or pressed_key & 0xFF == ord('M'):
            points.clear()
            writing_mode = not writing_mode
            prediction_mode = not prediction_mode
        elif pressed_key & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
