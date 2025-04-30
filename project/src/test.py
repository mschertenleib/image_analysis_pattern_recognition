import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("data/project/train/L1000957.JPG")

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.resize(gray, dsize=None, fx=1 / 16, fy=1 / 16, interpolation=cv2.INTER_LINEAR)

grad_x = cv2.Sobel(
    gray,
    ddepth=cv2.CV_32F,
    dx=1,
    dy=0,
    ksize=5,
    borderType=cv2.BORDER_REPLICATE,
)
grad_y = cv2.Sobel(
    gray,
    ddepth=cv2.CV_32F,
    dx=0,
    dy=1,
    ksize=5,
    borderType=cv2.BORDER_REPLICATE,
)
grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
plt.imshow(grad)
plt.show()
