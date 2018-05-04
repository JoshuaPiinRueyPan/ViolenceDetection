import src.third_party.imageAugmentation.imgaug as libRoot
from src.third_party.imageAugmentation.imgaug import augmenters as lib
import time
import numpy as np
import settings.DataSettings as dataSettings

def _augmentedByAllMethods():
	sometimes = lambda aug: lib.Sometimes(0.5, aug)
	augmentMethod = lib.Sequential([
					# horizontally flip 50% of all images
					lib.Fliplr(0.5),

					# crop images by -5% to 10% of their height/width
					sometimes( lib.CropAndPad(
								    percent=(-0.05, 0.1),
								    pad_mode=libRoot.ALL,
								    pad_cval=(0, 255)
					)),
					sometimes(lib.Affine(
								# scale images to 80-120% of their size, individually per axis
								scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
 								# translate by -20 to +20 percent (per axis)
								translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
								# rotate by -15 to +15 degrees
								rotate=(-15, 15),
								# shear by -16 to +16 degrees
								shear=(-16, 16),
								# use nearest neighbour or bilinear interpolation (fast)
								order=[0, 1],
 								# if mode is constant, use a cval between 0 and 255
								cval=(0, 255),
								# use any of scikit-image's warping modes
								mode=libRoot.ALL
					)),
					# execute 0 to 5 of the following (less important) augmenters per image
					# don't execute all of them, as that would often be way too strong
					lib.SomeOf( (0, 5),
						    [
 							# convert images into their superpixel representation
							sometimes( lib.Superpixels(p_replace=(0, 1.0),
										   n_segments=(20, 200))
							),
							lib.OneOf([
									# blur images with a sigma between 0 and 3.0
									lib.GaussianBlur((0, 3.0)),
									# blur image using local means with kernel sizes
									# between 2 and 7
									lib.AverageBlur(k=(2, 7)),
									# blur image using local medians with kernel sizes
									# between 2 and 7
									lib.MedianBlur(k=(3, 11)),
							]),
							# sharpen images
							lib.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
							# emboss images
							lib.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
							# search either for all edges or for directed edges,
							# blend the result with the original image using a blobby mask
							lib.SimplexNoiseAlpha(lib.OneOf([
											  lib.EdgeDetect(alpha=(0.5, 1.0)),
											  lib.DirectedEdgeDetect(alpha=(0.5, 1.0),
											  direction=(0.0, 1.0)),
							])),
							# add gaussian noise to images
							lib.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
							lib.OneOf([
								    # randomly remove up to 10% of the pixels
								    lib.Dropout((0.01, 0.1), per_channel=0.5),
								    lib.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05),
										      per_channel=0.2),
							]),
							# invert color channels
							lib.Invert(0.05, per_channel=True),
							# change brightness of images (by -10 to 10 of original value)
							lib.Add((-10, 10), per_channel=0.5),
							# change hue and saturation
							lib.AddToHueAndSaturation((-20, 20)),
							# either change the brightness of the whole image (sometimes
							# per channel) or change the brightness of subareas
							lib.OneOf([
									lib.Multiply((0.5, 1.5), per_channel=0.5),
									lib.FrequencyNoiseAlpha(
										exponent=(-4, 0),
										first=lib.Multiply((0.5, 1.5), per_channel=True),
										second=lib.ContrastNormalization((0.5, 2.0))
									)
							]),
							# improve or worsen the contrast
							lib.ContrastNormalization((0.5, 2.0), per_channel=0.5),
							lib.Grayscale(alpha=(0.0, 1.0)),
							# move pixels locally around (with random strengths)
							sometimes(lib.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
							# sometimes move parts of the image around
							sometimes(lib.PiecewiseAffine(scale=(0.01, 0.05))),
							sometimes(lib.PerspectiveTransform(scale=(0.01, 0.1)))
						    ],
						    random_order=True
						  )
					],
					random_order=True
				      )
	return augmentMethod

def _augmentedBySelectedMethods():
	sometimes = lambda aug: lib.Sometimes(0.5, aug)
	augmentMethod = lib.Sequential([
						# horizontally flip 50% of all images
						lib.Fliplr(0.5),

						# execute 0 to 5 of the following (less important) augmenters per image
						# don't execute all of them, as that would often be way too strong
						lib.SomeOf( (0, 5),
							    [
								lib.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
								lib.Emboss(alpha=(0, 1.0), strength=(0, 1.0)),
								# search either for all edges or for directed edges,
								# blend the result with the original image using a blobby mask
								lib.SimplexNoiseAlpha(lib.OneOf([
								    lib.EdgeDetect(alpha=(0.3, 0.6)),
								    lib.DirectedEdgeDetect(alpha=(0.3, 0.6), direction=(0.0, 1.0)),
								])),
								# add gaussian noise to images
								lib.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),
												 per_channel=0.5),

								lib.Dropout( (0.0, 0.05), per_channel=0.5),

								# change brightness of images (by -10 to 10 of original value)
								lib.Add((-10, 10), per_channel=0.5),

								# change hue and saturation
								lib.AddToHueAndSaturation((-20, 20)),

								# either change the brightness of the whole image (sometimes
								# per channel) or change the brightness of subareas
								lib.OneOf([
								    lib.Multiply((0.5, 1.5), per_channel=0.5),
								    lib.FrequencyNoiseAlpha(
									exponent=(-4, 0),
									first=lib.Multiply((0.5, 1.5), per_channel=True),
									second=lib.ContrastNormalization((0.5, 2.0))
								    )
								]),
								# improve or worsen the contrast
								lib.ContrastNormalization((0.6, 2.0), per_channel=0.5),
								lib.Grayscale(alpha=(0.0, 0.8)),
								# sometimes move parts of the image around
								sometimes(lib.PerspectiveTransform(scale=(0.01, 0.1)))
							    ],
							    random_order=True
							)
					],
					random_order=True
				      )

	return augmentMethod



'''
    The 3-party data augmentation seems has data race issue.
    Job with 3-party data aug. will turn into silently dead
    after several epochs.  Here, try to Sync data augs.
    in every thread.
'''
def Augment(batchOfImagesInUINT8_):
	'''
	    'images' should be either a 4D numpy array of shape (N, height, width, channels)
	    or a list of 3D numpy arrays, each having shape (height, width, channels).
	    All images must have numpy's dtype uint8. Values are expected to be in
	    range 0-255.
	'''
	startAugTime = time.time()
	augmentMethod = _augmentedBySelectedMethods()
	'''
	    The following Augmenter will augment images in the same way.
	'''
	deterministicAugmentMethod = augmentMethod.to_deterministic()

	batchOfResultImages = np.zeros(batchOfImagesInUINT8_.shape, dtype=batchOfImagesInUINT8_.dtype)
	for i in range(batchOfImagesInUINT8_.shape[0]):
		batchOfResultImages[i] = deterministicAugmentMethod.augment_image(batchOfImagesInUINT8_[i])

	endAugTime = time.time()
	return batchOfResultImages
