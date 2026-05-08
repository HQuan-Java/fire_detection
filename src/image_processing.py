import cv2
import numpy as np


class FireImageProcessor:
    def __init__(self):
        self.kernel = np.ones((5, 5), np.uint8)

    def detect_fire_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 120, 150])
        upper1 = np.array([15, 255, 255])

        lower2 = np.array([15, 100, 150])
        upper2 = np.array([35, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)

        area = cv2.countNonZero(mask)
        return mask, area