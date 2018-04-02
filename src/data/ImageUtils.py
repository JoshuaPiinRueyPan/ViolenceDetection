import cv2
import numpy as np

def ResizeAndPad(img, targetSize_, padColor=0):
	'''
	    The following method is Copy from:
		https://stackoverflow.com/questions/44720580/resize-image-canvas-to-maintain-square-aspect-ratio-in-python-opencv
	'''
	h, w = img.shape[:2]
	sh, sw = targetSize_

	# interpolation method
	if h > sh or w > sw: # shrinking image
		interp = cv2.INTER_AREA

	else: # stretching image
		interp = cv2.INTER_CUBIC

	# aspect ratio of image
	aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

	# compute scaling and pad sizing
	if aspect > 1: # horizontal image
		new_w = sw
		new_h = np.round(new_w/aspect).astype(int)
		pad_vert = (sh-new_h)/2
		pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
		pad_left, pad_right = 0, 0

	elif aspect < 1: # vertical image
		new_h = sh
		new_w = np.round(new_h*aspect).astype(int)
		pad_horz = (sw-new_w)/2
		pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
		pad_top, pad_bot = 0, 0

	else: # square image
		new_h, new_w = sh, sw
		pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

	# set pad color
	if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
		padColor = [padColor]*3

	# scale and pad
	scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
	scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
					borderType=cv2.BORDER_CONSTANT, value=padColor)

	return scaled_img

