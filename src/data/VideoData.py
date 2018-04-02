import skvideo.io
import numpy as np
import src.data.ImageUtils as imageUtil
import settings.DataSettings as dataSettings
import cv2

class VideoData:
	def __init__(self, PATH_NAME_TO_VIDEO_):
		self.name = PATH_NAME_TO_VIDEO_
		self.isValid = False
		self.hasLabel = False

		self._images = None
		self._labels = None
		self._loadVideoImages()

	def SetLabel(self, fightStartFrame_, fightEndFrame_):
		self._labels = np.zeros([self.totalFrames, 2])
		for frameIndex in range(self.totalFrames):
			if (frameIndex >= float(fightStartFrame_))and(frameIndex <= float(fightEndFrame_)):
				self._labels[frameIndex] = np.array( [0., 1.] )  # Fight
			else:
				self._labels[frameIndex] = np.array( [1., 0.] )  # None-Fight

		self.hasLabel = True

	@property
	def images(self):
		if self.isValid:
			return self._images

		else:
			raise ValueError("Video has no images! Please check: '" + self.name + "'\n"
					 + "\t Note: You can call VideoData.isValid, "
					 + "to check if the video has any frame.")
	@property
	def labels(self):
		if self.hasLabel:
			return self._labels
		else:
			raise ValueError("Video has no labels!\n"
					 + "\t Note: You can call VideoData.hasLabel, "
					 + "to check if the video has ground truth.")


	@property
	def totalFrames(self):
		if self.isValid:
			return self._images.shape[0]

		else:
			return 0


	def _loadVideoImages(self):
		try:
			rgbImages = skvideo.io.vread(self.name)
			self._images = self._convertRawImageToNetInput(rgbImages)

			self.isValid = True

		except Exception as error:
			print("---------------------------------------------")
			print("Video: " + self.name)
			print(error)
			print("ignore the video because of the above error...")
			print("---------------------------------------------")
			self.isValid = False

	def _convertRawImageToNetInput(self, rgbImages_):
		numberOfImages = rgbImages_.shape[0]
		enlargedImages = np.zeros([numberOfImages, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, 3])
		for i in range(numberOfImages):
			enlargedImages[i] = imageUtil.ResizeAndPad(rgbImages_[i], (dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE) )

		netInputImages = (enlargedImages/255.0) * 2.0 - 1.0
		return netInputImages
	

