import random
from abc import ABCMeta, abstractmethod
import cv2
import settings.DataSettings as dataSettings

class DataManagerBase(object):
	__metaclass__ = ABCMeta
	def __init__(self, LIST_OF_DATA_INFO_):
		self._listOfData = []
		self._loadAllVideosMemory(LIST_OF_DATA_INFO_)

	def _loadAllVideosMemory(self, LIST_OF_DATA_INFO_):
		'''
		    The data are expected in the following format:
		'''
		for eachDataInfo in LIST_OF_DATA_INFO_:
			pathToVideo, fightStartFrame, fightEndFrame = eachDataPath.split('\t')
			currentVideoReader = cv2.VideoCapture(pathToVideo)

			if currentVideoReader.isOpened():
				currentLabel = (fightStartFrame, fightEndFrame)
				self._listOfData.append( (currentVideoReader, currentLabel) )
			
			else:
				print("Unable to open: " + videoPathName)

	
	def _assignDataFromSingleVideo(self, video_, currentLabel_,
					 frameStartIndex_, NUMBER_OF_FRAMES_TO_CONCAT,
					 arrayToAssignImages_, arrayToAssignLabels_):
		'''
		    Note for cv2.VideoCapture:
			1. cv2.VideoCapture.frame start from 1
			2. When calling video.set(..., 5), to change the frame position to 5,
			   it will not return image unless the next time you call 'video.read()'.
			3. If one call 'video.set(..., 5)' then call 'video.read()', you will
			   actually get the 6th frame.
			4. The 'totalFrames' from 'video.get(cv2.CAP_PROP_FRAME_COUNT)'
			   is accessible.
		'''
		video_.set(cv2.CAP_PROP_POS_FRAMES, frameStartIndex_)
		fightStartFrame, fightEndFrame = currentLabel_

		isCurrentFrameValid, bgrImage = videoReader.read()

		if not isCurrentFrameValid:
			raise ValueError("Can't read Video, do you remove video at run time?")

		rgbImage = cv2.cvtColor(bgrImage, cv2.BGR2RBG);
		rgbImage /= 255.

		for arrayIndex in range(NUMBER_OF_FRAMES_TO_CONCAT):
			if isCurrentFrameValid:
				arrayToAssignImages_[arrayIndex] = rgbImage
				currentFramePosition = frameStartIndex_ + 1 + arrayIndex
				if (currentFramePosition >= fightStartFrame)and(currentFramePosition <= fightEndFrame):
					arrayToAssignLabels_[arrayIndex] = np.array([0., 1.])
				else:
					arrayToAssignLabels_[arrayIndex] = np.array([1., 0.])

			else:
				'''
				    For the case that UNROLLED_SIZE > video.TOTAL_FRAMES,
				    use the last frame always.
				'''
				arrayToAssignImages_[arrayIndex] = arrayToAssignImages_[arrayIndex-1]
				arrayToAssignLabels_[arrayIndex] = arrayToAssignLabels_[arrayIndex-1]

			isCurrentFrameValid, currentImage = videoReader.read()

	@abstractmethod
	def GetBatchOfData(self):
		pass


class TrainDataManager(DataManagerBase):
	def __init__(self):
		super(DataManagerBase, self).__init__(dataSettings.PATH_TO_TRAIN_SET_LIST)
		self._isNewEpoch = True
		self._dataCursor = 0
		self.epoch = 0
		self.step = 0

	def GetBatchOfData(self):
		self._isNewEpoch = False
		arrayOfImages = np.zeros(dataSettings.BATCH_SIZE, dataSettings.UNROLLED_SIZE,
					 dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, 3)
		arrayOfLabels = np.zeros(dataSettings.BATCH_SIZE, dataSettings.UNROLLED_SIZE, 2)

		outputIndex = 0
		while outputIndex < dataSettings.BATCH_SIZE:
			currentVideo, currentLabel = self._listOfData[self._dataCursor]
			totalFrames = currentVideo.get(cv2.CAP_PROP_FRAME_COUNT)
			frameStartIndex = random.randint(0, max(0, totalFrames - dataSettings.UNROLLED_SIZE) )
			_assignDataFromSingleVideo(currentVideo, currentLabel,
						   frameStartIndex, dataSettings.UNROLLED_SIZE,
						   arrayOfImages[outputIndex], arrayOfLabels[outputIndex])
			outputIndex += 1
			self._dataCursor += 1
			if self._dataCursor >= len(self._listOfData):
				shuffle(self._listOfData)
				self._dataCursor = 0
				self.epoch += 1
				self.isNewEpoch = True
		self.step += 1

		arrayOfImages = arrayOfImages.reshape([-1, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, 3])
		arrayOfLabels = arrayOfLabels.reshape([-1, 2])
		return arrayOfImages, arrayOfLabels


class EvaluationDataManager(DataManagerBase):
	'''
	    This DataManager is design for Validation & Test.
	    Different from TrainDataManager, EvaluationDataManager
	    will try to pach the Same Video into a batch.  And if
	    there're more space, this manager will not keep packing
	    images from other video.

	    Usage:
		def CalculateValidation():
			valDataSet = EvaluationDataManager("./val.txt")

			valLoss = 0
			while not valDataSet.isAllDataTraversed:
				valLoss += net.CalculateLoss(valDataSet.GetBatchOfData())
				if valDataSet.isNewVideo:
					net.ResetCellState()
	'''
	def __init__(self, PATH_TO_DATA_SET_LIST_):
		super(DataManagerBase, self).__init__(PATH_TO_DATA_SET_LIST_)
		self.isAllDataTraversed = False
		self.isNewVideo = True
		self._videoCursor = 0
		self._frameCursor = 0

	def GetBatchOfData(self):
		self.isAllDataTraversed = False
		self.isNewVideo = False
		currentVideo, currentLabel = self._listOfData[self._videoCursor]
		totalFrames = currentVideo.get(cv2.CAP_PROP_FRAME_COUNT)

		if totalFrames < 1:
			raise ValueError("Video has no frame, please check...")

		unrolledSize = min(dataSettings.BATCH_SIZE * dataSettings.UNROLLED_SIZE,
				   totalFrames - self._frameCursor)

		arrayOfImages = np.zeros(unrolledSize, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, 3)
		arrayOfLabels = np.zeros(unrolledSize, 2)

		_assignDataFromSingleVideo(currentVideo, currentLabel,
					   self._frameCursor, unrolledSize,
					   arrayOfImages, arrayOfLabels)

		self._frameCursor += unrolledSize
		if self._frameCursor >= totalFrames:
			self._videoCursor +=1
			self.isNewVideo = True
			if self._videoCursor >= len(self._listOfData):
				self._videoCursor = 0
				self.isAllDataTraversed = True
		

		return arrayOfImages, arrayOfLabels

