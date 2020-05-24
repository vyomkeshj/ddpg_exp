import numpy as np
from timeit import default_timer as timer
import cv2

image = np.zeros((1080,1920,3),np.uint8)
buff = np.ones((1080,1920),np.uint8)*550
try:
	while(True):
		now = timer()
		x_start = 500;
		y_start = 200;
		x_end = 600;
		y_end = 400
		x_values = np.linspace(start = x_start, stop = x_end-1, num=x_end-x_start, dtype=int);
		y_values = np.linspace(start = y_start, stop = y_end-1, num=y_end-y_start, dtype=int);
		mask = buff[y_values, x_values]>x_values;
		print((timer()- now) * 1000)
		cv2.imshow("img", image)
		cv2.waitKey(1)


except KeyboardInterrupt:
	exit()
