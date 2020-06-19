import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from gpiozero import LED


class VideoCamera(object):
    def __init__(self):
        self.milliseconds = 0
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file instead.

        # raspi pins
        self.unlock = LED(17)
        self.lock = LED(18)

        # open the default camera
        self.camera = cv.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv.VideoCapture('video.mp4')

        # this enables facial detection for OpenCV
        self.face_detection = cv.CascadeClassifier(
            'haarcascade_frontalface_default.xml')

        # loads model
        self.model = tf.keras.models.load_model('mask_detection_model_v7')

        # 0 is mask, 1 is no mask
        self.labels_dict = {0: 'MASK', 1: 'NO MASK'}
        # green box, red box
        self.color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

    def __del__(self):
        self.camera.release()

    def get_frame(self):
        self.milliseconds = self.milliseconds + 1

        # camera.read() returns 2 variables, one matrix of the image one video
        matrix, frame = self.camera.read()

        # the video in tensor or matrix form
        # print(matrix)
        # the actual video stream
        # print(frame)

        # convert frame into grey scale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # get faces from gray video frame
        faces = self.face_detection.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # crops the region which the face is found within the gray frame
            face_img = gray[y:y+w, x:x+w]
            # resizing image to what the nn was trained for
            # cv.resize(face_img,(50,50) returns a tensor of the image
            resized = cv.resize(face_img, (100, 100))

            # dividing image by 255 because each pixel only has a max value of 255
            # so by doing this each value returned in the resize tensor will now be
            # between 0-1 making it easier for our nn to manage
            normalized = resized/255.0
            # reshapes size of array to 4D since convnets takes in 4d array
            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            # images shown in tensors
            # print(reshaped)
            # how many array dimensions
            # print(reshaped.ndim)

            # shows the image that is sent to the nn
            # plt.imshow(resized,cmap="gray")
            # plt.show()

            # runs image of face that was caputures through the model
            result = self.model.predict(reshaped)
            print(result)
            # returns index of mask status
            label = np.argmax(result, axis=1)[0]
            print(self.labels_dict[label], " ", result[0][label]*100)

            # if(int(result[0][0]*100) < 97):
            #     label = 1

            if(label == 0):
                self.unlock.on()
                self.lock.off()
            else:
                self.unlock.off()
                self.lock.on()

            cv.rectangle(frame, (x, y), (x+w, y+h), self.color_dict[label], 2)
            cv.rectangle(frame, (x, y-40), (x+w, y),
                         self.color_dict[label], -1)
            cv.putText(frame, self.labels_dict[label], (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(frame, str(round(
                result[0][0]*100, 2)), (x+125, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv.imencode('.jpg', frame)
        return jpeg.tobytes()
