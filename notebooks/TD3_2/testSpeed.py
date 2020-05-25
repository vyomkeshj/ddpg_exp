import numpy as np
from timeit import default_timer as timer

image = np.zeros((1080,1920,3),np.uint8)
buff = np.ones((1080,1920),np.uint8)*550
try:
	while(True):
		image = np.zeros((1080, 1920, 3), np.uint8)
		now = timer()

		for y in range(200, 400):
			for x in range(500, 600):


				#image[y,x,0] = 255
				if buff[y, x] > x:
					image[y, x, 0] = 255		#todo: C[i,j, K]  = 255
				else:
					image[y, x, 1] = 255
		print((timer()- now) * 1000)







except KeyboardInterrupt:
	exit()
