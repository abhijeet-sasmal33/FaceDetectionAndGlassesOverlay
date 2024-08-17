import cv2
import numpy as np
import math


def create_rectangle(img, x, y, x1, y1, color=(0, 255, 0), thickness=2):
    cv2.rectangle(img, (x, y), (x1, y1), color, thickness)
    return img


def image_to_polygons(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    return thresh


face_cascade = cv2.CascadeClassifier("harcascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("harcascade/haarcascade_eye_tree_eyeglasses.xml")


glasses_clr = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    back = frame

    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        eyes_frame = frame
        eyes_frame = create_rectangle(
            img=eyes_frame, x=x, y=y, x1=x + w, y1=y + h, color=(255, 0, 0)
        )

        try:
            ex, ey, ew, eh = eyes[0]
            ex1, ey1, ew1, eh1 = eyes[1]

            glasses = image_to_polygons("glasses.png")

            roi = eyes_frame[ey + y : ey1 + ew1 + y, ex + x - 10 : ex1 + eh1 + x + 10]
            y, x, z = roi.shape
            resized_glassses = cv2.resize(glasses, (x, y))

            roi[np.where(resized_glassses, False, True)] = 0

        except Exception as e:
            pass

        cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
