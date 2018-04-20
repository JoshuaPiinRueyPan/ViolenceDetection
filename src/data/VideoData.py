import skvideo.io
import numpy as np
import src.data.ImageUtils as ImageUtils
import settings.DataSettings as dataSettings
import cv2

class VideoData:
	def __init__(self, PATH_NAME_TO_VIDEO_, fightStartFrame_, fightEndFrame_):
		self.name = PATH_NAME_TO_VIDEO_
		self.hasImages = False
		self.hasLabel = False
		self.totalFrames = 0

		self._images = None
		self._labels = None
		self._peekVideoTotalFrames()
		self._calculateLabels(fightStartFrame_, fightEndFrame_)

	def _peekVideoTotalFrames(self):
		videoReader = cv2.VideoCapture(self.name)
		self.totalFrames = videoReader.get(cv2.CAP_PROP_FRAME_COUNT)
		if self.totalFrames.is_integer():
			self.totalFrames = int(self.totalFrames)
		else:
			errorMessage = "Video: " + self.name
			errorMessage += " has totalFrames(=" + str(self.totalFrames) + ")"
			errorMessage += "  is not an integer, please check!"
			raise ValueError(errorMessage)
		videoReader.release()
		

	def _calculateLabels(self, fightStartFrame_, fightEndFrame_):
		self._labels = np.zeros( [int(self.totalFrames), 2] )
		for frameIndex in range(self.totalFrames):
			if (frameIndex >= float(fightStartFrame_))and(frameIndex <= float(fightEndFrame_)):
				self._labels[frameIndex] = np.array( dataSettings.FIGHT_LABEL )  # Fight
			else:
				self._labels[frameIndex] = np.array( dataSettings.NO_FIGHT_LABEL )  # None-Fight

		self.hasLabel = True

	@property
	def images(self):
		'''
		    Note: After you finish the use of images, you may want to release the
		    images by: VideoData.images = None
		'''
		if self.hasImages:
			return self._images
		else:
			raise ValueError("No image found in video: " + self.name + ",\n"
					 + "Do you forget to call 'VideoData.LoadVideoImages()'"
					 + " before you try to access the images?  or is the video broken?")



	@property
	def labels(self):
		if self.hasLabel:
			return self._labels
		else:
			raise ValueError("Video has no labels!\n"
					 + "\t Note: You can call VideoData.hasLabel, "
					 + "to check if the video has ground truth.")

	def LoadVideoImages(self, dataAugmentFunction_=None):
		'''
		    This function will Block the current thread utill the images are loaded.
		'''
		try:
			rgbImages = skvideo.io.vread(self.name)
			numberOfLoadedImages = rgbImages.shape[0]
			if self.totalFrames != numberOfLoadedImages:
				print("Warning! self.totalFrames (="+str(self.totalFrames)+") != loadedImages(="
					+ str(numberOfLoadedImages) + ")!")
				print("\t This may due to the inconsistence of OpenCV & Sk-Video...")
				self.totalFrames = numberOfLoadedImages
				self._calculateLabels()

			if dataAugmentFunction_ != None:
				rgbImages = dataAugmentFunction_(rgbImages)

			self._images = np.zeros([numberOfLoadedImages,
						 dataSettings.IMAGE_SIZE,
						 dataSettings.IMAGE_SIZE,
						 dataSettings.IMAGE_CHANNELS])
			for i in range(numberOfLoadedImages):
				self._images[i] = ImageUtils.ConvertImageFrom_RGB255_to_NetInput(rgbImages[i])

			self.hasImages = True

		except Exception as error:
			print("---------------------------------------------")
			print("Video: " + self.name)
			print(error)
			print("ignore the video because of the above error...")
			print("---------------------------------------------")
			self.hasImages = False

	def ReleaseImages(self):
		self._images = None

