import tensorflow as tf
import numpy as np

def ResizeAndPad(images_, TARGET_SIZE_):
	'''
	    Arguments:
		  images_: Should be an 4-D Tensor or 4-D np.array.
			   ex: [batchSize, w, h, c]
		  TARGET_SIZE_: Should be a Value represent the target
				width & heights.  Note:The target image
				must be square image.
	    Note: This function will first pad the images to square,
		  and enlarge it to the target size.
	'''
	maxLength = max(images_.shape[1], images_.shape[2])
	squaredImages =  tf.image.resize_image_with_crop_or_pad(images_, maxLength, maxLength)
	enlargedImages = tf.image.resize_image(squaredImages, [TARGET_SIZE_, TARGET_SIZE_])

	return enlargedImages
