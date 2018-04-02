import random
from abc import ABCMeta, abstractmethod
import settings.DataSettings as dataSettings
from src.data.VideoData import VideoData
import numpy as np
import random

class DataManagerBase:
	__metaclass__ = ABCMeta
	def __init__(self, PATH_TO_DATA_SET_CATELOG_):
		self._listOfData = []
		self._loadAllVideosMemory(PATH_TO_DATA_SET_CATELOG_)

	def _loadAllVideosMemory(self, PATH_TO_DATA_SET_CATELOG_):
		'''
		    The data are expected in the following format:
		'''
		with open(PATH_TO_DATA_SET_CATELOG_, 'r') as fileContext:
			for eachLine in fileContext:
				pathToVideo, fightStartFrame, fightEndFrame = eachLine.split('\t')
				currentVideo = VideoData(pathToVideo)
				currentVideo.SetLabel(fightStartFrame, fightEndFrame)

				if currentVideo.isValid and currentVideo.hasLabel:
					self._listOfData.append( currentVideo )
				
				else:
					print("Unable to open: " + pathToVideo)


	def _getDataFromSingleVideo(self, video_, startFrameIndex_, NUMBER_OF_FRAMES_TO_CONCAT_):
		endFrameIndex = startFrameIndex_ + NUMBER_OF_FRAMES_TO_CONCAT_
		if endFrameIndex < video_.totalFrames:
			arrayOfImages = video_.images[startFrameIndex_ : endFrameIndex]
			arrayOfLabels = video_.labels[startFrameIndex_ : endFrameIndex]
			return arrayOfImages, arrayOfLabels

		else:
			'''
			    For the case that UNROLLED_SIZE > video.TOTAL_FRAMES,
			    use the last frame always.
			'''
			listOfImages = []
			listOfLabels = []
			
			listOfImages.append(video_.images[startFrameIndex_:])
			listOfLabels.append(video_.labels[startFrameIndex_:])

			numberOfArtificialFrames = endFrameIndex - video_.totalFrames

			while len(listOfImages) < numberOfArtificialFrames:
				listOfImages.append( [ video_.images[-1] ] )
				listOfLabels.append( [ video_.labels[-1] ] )

			arrayOfImages = np.concatenate( listOfImages, axis=0 )
			arrayOfLabels = np.concatenate( listOfLabels, axis=0 )

			return arrayOfImages, arrayOfLabels


	@abstractmethod
	def GetBatchOfData(self):
		pass


class TrainDataManager(DataManagerBase):
	def __init__(self):
		super().__init__(dataSettings.PATH_TO_TRAIN_SET_LIST)
		self._isNewEpoch = True
		self._dataCursor = 0
		self.epoch = 0
		self.step = 0

	def GetBatchOfData(self):
		self._isNewEpoch = False
		listOfBatchImages = []
		listOfBatchLabels = []

		outputIndex = 0
		while outputIndex < dataSettings.BATCH_SIZE:
			currentVideo = self._listOfData[self._dataCursor]
			frameStartIndex = random.randint(0, max(0, currentVideo.totalFrames - dataSettings.UNROLLED_SIZE) )
			arrayOfImages, arrayOfLabels = self._getDataFromSingleVideo(currentVideo,
										    frameStartIndex, dataSettings.UNROLLED_SIZE)
			listOfBatchImages.append(arrayOfImages)
			listOfBatchLabels.append(arrayOfLabels)
			outputIndex += 1
			self._dataCursor += 1
			if self._dataCursor >= len(self._listOfData):
				random.shuffle(self._listOfData)
				self._dataCursor = 0
				self.epoch += 1
				self.isNewEpoch = True
		self.step += 1

		arrayOfBatchImages = np.concatenate(listOfBatchImages, axis=0)
		arrayOfBatchLabels = np.concatenate(listOfBatchLabels, axis=0)
		return arrayOfBatchImages, arrayOfBatchLabels


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
	def __init__(self, PATH_TO_DATA_SET_CATELOG_):
		super().__init__(PATH_TO_DATA_SET_CATELOG_)
		self.isAllDataTraversed = False
		self.isNewVideo = True
		self._videoCursor = 0
		self._frameCursor = 0

	def GetBatchOfData(self):
		self.isAllDataTraversed = False
		self.isNewVideo = False
		currentVideo = self._listOfData[self._videoCursor]

		unrolledSize = min(dataSettings.BATCH_SIZE * dataSettings.UNROLLED_SIZE,
				   currentVideo.totalFrames - self._frameCursor)

		arrayOfImages, arrayOfLabels = self._getDataFromSingleVideo(currentVideo,
									    self._frameCursor, unrolledSize)

		self._frameCursor += unrolledSize
		if self._frameCursor >= currentVideo.totalFrames:
			self._frameCursor = 0
			self._videoCursor +=1
			self.isNewVideo = True
			if self._videoCursor >= len(self._listOfData):
				self._videoCursor = 0
				self.isAllDataTraversed = True
		

		return arrayOfImages, arrayOfLabels

