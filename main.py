import cv2
import numpy as np
import pyautogui
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np




cap = cv2.VideoCapture(1)

lower_g = np.array([20, 100, 100])  # yellow
upper_g = np.array([30, 255, 255])  # yellow

lower_r = np.array([100, 150, 0])  # red
upper_r = np.array([140, 255, 255])  # red

# frame = np.zeros((512,512,3), np.uint8)
x, y, k = 200, 200, -1
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
center_x_pre_y = 0
center_y_prev_y = 0
center_x1_y = 1
center_y1_y = 1

center_x_pre_r = 0
center_y_prev_r = 0
center_x1_r = 1
center_y1_r = 1


def show(frame):
    cv2.imshow("digit", frame)


while True:
    r, frame2 = cap.read()
    cv2.imshow("new_window", frame2)

    if k == 1 or cv2.waitKey(10) == ord("q"):
        cv2.destroyAllWindows()
        break

new = np.zeros_like(frame2)

while True:
    r, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_r, upper_r)
    mask2 = cv2.inRange(hsv, lower_g, upper_g)
    image, contours1, h = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image, contours2, h = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours1:
        area = cv2.contourArea(c)
        if (area > 100):
            x, y, w, h = cv2.boundingRect(c)
            center_x1_y = int(x + (w / 2))
            center_y1_y = int(y + (h / 2))
            cv2.circle(frame, (center_x1_y, center_y1_y), 10, (255, 255, 0), 5)

    for c in contours2:
        area = cv2.contourArea(c)
        if (area > 100):
            x, y, w, h = cv2.boundingRect(c)
            center_x1_r = int(x + (w / 2))
            center_y1_r = int(y + (h / 2))
            cv2.circle(frame, (center_x1_r, center_y1_r), 10, (0, 153, 0), 5)

    new = cv2.line(new, (center_x_pre_y, center_y_prev_y), (center_x1_y, center_y1_y), (255, 0, 0), 4)
    center_x_pre_y = center_x1_y
    center_y_prev_y = center_y1_y

    new = cv2.line(new, (center_x_pre_r, center_y_prev_r), (center_x1_r, center_y1_r), (0, 0, 255), 4)
    center_x_pre_r = center_x1_r
    center_y_prev_r = center_y1_r

    new = cv2.addWeighted(new, 0.5, new, 0.5, 0)

    cv2.imshow('frame', frame)
    cv2.imshow("final", new)

    if k == 1 or cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(10) == ord("n"):
        new = new = np.zeros_like(frame2)

    if cv2.waitKey(10) == ord("c"):
        cv2.imshow("Image", new)
        show(new)

cap.release()

