from abc import ABCMeta, abstractmethod
import random
import numpy as np
import time
import threading
import settings.TrainSettings as trainSettings
import settings.EvaluationSettings as evalSettings
import settings.DataSettings as dataSettings
from src.data.VideoData import VideoData
import six
if six.PY2:
	from Queue import *
else:
	from queue import *

'''
    When user call DataManager.Stop(), the loading threads
    may keep runing because of the Blocked Queue.get()
    or Blocked Queue.put(), thus add the following timeout.
'''
TIMEOUT_FOR_WAIT_QUEUE = 10
'''
   Number of Data to Produce when One data is Consumed.
'''
PRODUCE_CONSUME_RATIO = 5

class BatchData:
	def __init__(self):
		self.batchSize = 0
		self.unrolledSize = 0
		self.batchOfImages = None
		self.batchOfLabels = None

class DataManagerBase:
	__metaclass__ = ABCMeta
	def __init__(self, PATH_TO_DATA_SET_CATELOG_):
		self._listOfData = []
		self._initVideoData(PATH_TO_DATA_SET_CATELOG_)
		self._lockForThreadControl = threading.Lock()
		self._shouldPause = False
		self._shouldStop = False
		self._listOfLoadDataThreads = []

		self._lockForDataList = threading.Lock()
		self._queueForWaitingVideos = Queue(maxsize=dataSettings.DATA_QUEUE_MAX_SIZE*2)
		self._queueForLoadedVideos = Queue(maxsize=dataSettings.DATA_QUEUE_MAX_SIZE)

		self.TOTAL_DATA = len(self._listOfData)

	def __del__(self):
		self.Stop()
		for eachThread in self._listOfLoadDataThreads:
			eachThread.join()
		print("\t TrainDataManager.thread.join() successfully.")

	def Pause(self):
		with self._lockForThreadControl:
			self._shouldPause = True

	def Continue(self):
		with self._lockForThreadControl:
			self._shouldPause = False

	def Stop(self):
		with self._lockForThreadControl:
			self._shouldStop = True

	def GetQueueInfo(self):
		info = "listOfData.len() = " + str( len(self._listOfData) ) + ";\t"
		info += "queueForWaiting.len() = " + str( self._queueForWaitingVideos.qsize() ) + ";\t"
		info += "queueForLoaded.len() = " + str(self._queueForLoadedVideos.qsize() ) + ";\t"
		with self._lockForThreadControl:
			info += "Pause = " + str(self._shouldPause)
		return info


	def _initVideoData(self, PATH_TO_DATA_SET_CATELOG_):
		'''
		    The data are expected in the following format:
		'''
		with open(PATH_TO_DATA_SET_CATELOG_, 'r') as fileContext:
			for eachLine in fileContext:
				try:
					pathToVideo, fightStartFrame, fightEndFrame = eachLine.split('\t')
					currentVideo = VideoData(pathToVideo, fightStartFrame, fightEndFrame)

					self._listOfData.append( currentVideo )

				except Exception as error:
					print(error)
		if len(self._listOfData) == 0:
			raise ValueError("No Valid Data found in: " + PATH_TO_DATA_SET_CATELOG_)

	def _executeLoadDataThreads(self, NUMBER_OF_LOAD_DATA_THREADS_):
		for i in range(NUMBER_OF_LOAD_DATA_THREADS_):
			currentThread = threading.Thread(target=self.runLoadingThread)
			currentThread.start()
			self._listOfLoadDataThreads.append(currentThread)

	def runLoadingThread(self):
		shouldPause = False
		shouldStop = False
		with self._lockForThreadControl:
			shouldPause = self._shouldPause
			shouldStop = self._shouldStop

		while not shouldStop:
			if shouldPause:
				time.sleep(0.5)

			else:
				try:
					self._loadVideoData()

				except Empty:
					'''
					    Catch the Queue.Empty exception thrown
					    by waiting timeout.
					'''
					time.sleep(0.0005)

			# Update shouldStop flag
			with self._lockForThreadControl:
				shouldPause = self._shouldPause
				shouldStop = self._shouldStop

	def _getDataFromSingleVideo(self, video_, startFrameIndex_, NUMBER_OF_FRAMES_TO_CONCAT_):
		endFrameIndex = startFrameIndex_ + NUMBER_OF_FRAMES_TO_CONCAT_
		if endFrameIndex <= video_.totalFrames:
			arrayOfImages = video_.images[startFrameIndex_ : endFrameIndex]
			arrayOfLabels = video_.labels[startFrameIndex_ : endFrameIndex]
			return arrayOfImages, arrayOfLabels

		else:
			'''
			    For the case that UNROLLED_SIZE > video.TOTAL_FRAMES,
			    use the last frame always.
			'''
			arrayOfImages = np.zeros( [NUMBER_OF_FRAMES_TO_CONCAT_,
						   dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_CHANNELS],
						  dtype=dataSettings.FLOAT_TYPE )
			arrayOfLabels = np.zeros( [NUMBER_OF_FRAMES_TO_CONCAT_, 2],
						  dtype=dataSettings.FLOAT_TYPE )
			
			arrayOfImages[ : video_.totalFrames] = video_.images[startFrameIndex_:]
			arrayOfLabels[ : video_.totalFrames] = video_.labels[startFrameIndex_:]

			numberOfArtificialFrames = endFrameIndex - video_.totalFrames
			arrayOfLastFrameImages = np.tile( video_.images[-1], [numberOfArtificialFrames, 1, 1, 1] )
			arrayOfLastFrameLabels = np.tile( video_.labels[-1], [numberOfArtificialFrames, 1] )

			arrayOfImages[video_.totalFrames : ] = arrayOfLastFrameImages
			arrayOfLabels[video_.totalFrames : ] = arrayOfLastFrameLabels

			return arrayOfImages, arrayOfLabels

	def pushVideoDataToWaitingQueue(self, numberOfData_):
		'''
		    This function push 'numberOfData_' from the head of 'self._listOfData'
		    to the queue that wait for loading video images.
		    Note: If the '_queueForWaitingVideos' is full, ignore push.
		'''
		for i in range(numberOfData_):
			try:
				with self._lockForDataList:
					videoReader = self._listOfData.pop(0)
				self._queueForWaitingVideos.put(videoReader, block=False)

			except Full:
				with self._lockForDataList:
					self._listOfData.append(videoReader)

			except IndexError:
				'''
				    For the case that DATA_QUEUE_MAX_SIZE > TOTAL_DATA,
				    the IndexError may be raised from '_listOfData.pop()'.
				'''
				pass



	def appendVideoDataBackToDataList(self, listOfVideoData_):
		'''
		    After you get the video from 'self._queueForLoadedVideos'
		    and perform some operation on the videos, you should stuff that
		    VideoReader back to the 'self._listOfData'.  Otherwise the
		    VideoReader will getting fewer and fewer.
		'''
		with self._lockForDataList:
			self._listOfData += listOfVideoData_


	@abstractmethod
	def AssignBatchData(self):
		pass

	@abstractmethod
	def _loadVideoData(self):
		pass


class TrainDataManager(DataManagerBase):
	def __init__(self, PATH_TO_DATA_SET_CATELOG_, NUMBER_OF_LOAD_DATA_THREADS=4):
		super().__init__(PATH_TO_DATA_SET_CATELOG_)

		# Public variables
		self._epoch = 0
		self._step = 0

		self._isNewEpoch = True
		self._dataCursor = 0

		self.pushVideoDataToWaitingQueue(dataSettings.DATA_QUEUE_MAX_SIZE*2)

		self._executeLoadDataThreads(NUMBER_OF_LOAD_DATA_THREADS)

	def AssignBatchData(self, batchData_):
		'''
		      The user should pass BatchData as argument to this function,
		    since this would be faster then this function return two numpy.array.

		      The 'batchData_.batchOfImages' will be assigned as the shape:
		    [BATCH_SIZE, UNROLLED_SIZE, w, h, c].
		      The 'batchData_.batchOfLabels' will be assigned as the shape:
		    [BATCH_SIZE, UNROLLED_SIZE, 2].
		'''
		self._isNewEpoch = False

		batchData = self._queueForLoadedVideos.get(block=True)

		batchData_.batchSize = batchData.batchSize
		batchData_.unrolledSize = batchData.unrolledSize
		batchData_.batchOfImages = batchData.batchOfImages
		batchData_.batchOfLabels = batchData.batchOfLabels

		self._step += 1
		self._dataCursor += trainSettings.BATCH_SIZE
		if self._dataCursor >= self.TOTAL_DATA:
			with self._lockForDataList:
				random.shuffle(self._listOfData)
			self._dataCursor = 0
			self._epoch += 1
			self._isNewEpoch = True

		self.pushVideoDataToWaitingQueue(trainSettings.BATCH_SIZE * PRODUCE_CONSUME_RATIO)

		return batchData

	@property
	def isNewEpoch(self):
		return self._isNewEpoch

	@property
	def epoch(self):
		return self._epoch

	@property
	def step(self):
		return self._step
	
	def _loadVideoData(self):
		if self._queueForLoadedVideos.qsize() <= dataSettings.DATA_QUEUE_MAX_SIZE:
			listOfLoadedVideos = []
			for i in range(trainSettings.BATCH_SIZE):
				videoReader = self._queueForWaitingVideos.get(block=True, timeout=TIMEOUT_FOR_WAIT_QUEUE)
				videoReader.LoadVideoImages()
				listOfLoadedVideos.append(videoReader)

			batchData = BatchData()
			batchData.batchSize = trainSettings.BATCH_SIZE
			batchData.unrolledSize = trainSettings.UNROLLED_SIZE

			arrayOfBatchImages = np.zeros( [batchData.batchSize, batchData.unrolledSize,
							dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_CHANNELS],
							dtype=dataSettings.FLOAT_TYPE )
			arrayOfBatchLabels = np.zeros( [batchData.batchSize, batchData.unrolledSize, 2],
							dtype=dataSettings.FLOAT_TYPE )

			for i in range(batchData.batchSize):
				currentVideo = listOfLoadedVideos[i]
				frameStartIndex = random.randint(0, 
								max(0, currentVideo.totalFrames - batchData.unrolledSize) )

				arrayOfImages, arrayOfLabels = self._getDataFromSingleVideo(currentVideo,
											    frameStartIndex,
											    batchData.unrolledSize)
				# Release the video frames
				currentVideo.ReleaseImages()

				arrayOfBatchImages[i] = arrayOfImages
				arrayOfBatchLabels[i] = arrayOfLabels

			batchData.batchOfImages = arrayOfBatchImages
			batchData.batchOfLabels = arrayOfBatchLabels

			try:
				self._queueForLoadedVideos.put(batchData, block=True, timeout=TIMEOUT_FOR_WAIT_QUEUE)
				self.appendVideoDataBackToDataList(listOfLoadedVideos)

			except Full:
				#print("\t\t LoadedQueue is full (size =", self._queueForLoadedVideos.qsize(),
				#      ");  put VideoReader back to WaitingQueue")
				try:
					while len(listOfLoadedVideos) > 0:
						eachVideoReader = listOfLoadedVideos.pop(0)
						self._queueForWaitingVideos.put(eachVideoReader, block=True,
										timeout=TIMEOUT_FOR_WAIT_QUEUE)
						#print("\t\t\t put to WaitingQueue (size = ", self._queueForWaitingVideos.qsize(),
						#      ")...")
				except Full:
					#print("\t\t\t WaitingQueue is full (size =", self._queueForWaitingVideos.qsize(),
					#      "); put VideoReader back to data list")
					listOfLoadedVideos.insert(0, eachVideoReader)
					with self._lockForDataList:
						self._listOfData = listOfLoadedVideos + self._listOfData

		
		else:
			time.sleep(0.001)



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
				valLoss += net.CalculateLoss(valDataSet.AssignBatchData())
				if valDataSet.isNewVideo:
					net.ResetCellState()

				if valDataSet.isAllDataTraversed:
					valDataSet.Pause()
	'''
	def __init__(self, PATH_TO_DATA_SET_CATELOG_, NUMBER_OF_LOAD_DATA_THREADS=1):
		super().__init__(PATH_TO_DATA_SET_CATELOG_)
		self._isAllDataTraversed = False
		self._isNewVideo = True
		self._dataCursor = 0
		self._currentVideo = None
		self._frameCursor = 0

		self._queueForWaitingVideos = Queue(maxsize=dataSettings.DATA_QUEUE_MAX_SIZE*2)
		self._queueForLoadedVideos = Queue(maxsize=dataSettings.DATA_QUEUE_MAX_SIZE)
		self.pushVideoDataToWaitingQueue(dataSettings.DATA_QUEUE_MAX_SIZE)

		self._executeLoadDataThreads(NUMBER_OF_LOAD_DATA_THREADS)

	def AssignBatchData(self, batchData_):
		'''
		      The user should pass BatchData as argument to this function,
		    since this would be faster then this function return two numpy.array.

		      The 'batchData_.batchOfImages' will be assigned as the shape:
		    [BATCH_SIZE, UNROLLED_SIZE, w, h, c].
		      The 'batchData_.batchOfLabels' will be assigned as the shape:
		    [BATCH_SIZE, UNROLLED_SIZE, 2].
		'''
		self._isAllDataTraversed = False
		self._isNewVideo = False
		if self._currentVideo == None:
			self._currentVideo = self._queueForLoadedVideos.get(block=True)

		unrolledSize = min(evalSettings.UNROLLED_SIZE,
				   self._currentVideo.totalFrames - self._frameCursor)

		batchData_.batchSize = 1
		batchData_.unrolledSize = unrolledSize
		batchData_.batchOfImages, batchData_.batchOfLabels = self._getDataFromSingleVideo(self._currentVideo,
												  self._frameCursor, unrolledSize)
		batchData_.batchOfImages = batchData_.batchOfImages.reshape([batchData_.batchSize,
									     batchData_.unrolledSize,
									     dataSettings.IMAGE_SIZE,
									     dataSettings.IMAGE_SIZE,
									     dataSettings.IMAGE_CHANNELS])
		batchData_.batchOfLabels = batchData_.batchOfLabels.reshape([batchData_.batchSize,
									     batchData_.unrolledSize,
									     dataSettings.NUMBER_OF_CATEGORIES])
		self._frameCursor += unrolledSize

		if self._frameCursor >= self._currentVideo.totalFrames:
			self._frameCursor = 0
			self._dataCursor += 1
			self._isNewVideo = True

			self.pushVideoDataToWaitingQueue(PRODUCE_CONSUME_RATIO)
			self.appendVideoDataBackToDataList( [self._currentVideo] )

			self._currentVideo.ReleaseImages()
			self._currentVideo = None
		
			if self._dataCursor >= self.TOTAL_DATA:
				self._dataCursor = 0
				self._isAllDataTraversed = True

	@property
	def isAllDataTraversed(self):
		return self._isAllDataTraversed

	@property
	def isNewVideo(self):
		return self._isNewVideo

	def _loadVideoData(self):
		if self._queueForLoadedVideos.qsize() <= dataSettings.DATA_QUEUE_MAX_SIZE:
			videoReader = self._queueForWaitingVideos.get(block=False)
			videoReader.LoadVideoImages()

			try:
				self._queueForLoadedVideos.put(videoReader, block=True, timeout=TIMEOUT_FOR_WAIT_QUEUE)

			except:
				videoReader.ReleaseImages()
				#print("\t\t LoadedQueue is full (size = ", self._queueForLoadedVideos.qsize(),
				#      "); stuff VideoReader back to WaitingQueue.")
				try:
					self._queueForWaitingVideos.put(videoReader, block=True, timeout=TIMEOUT_FOR_WAIT_QUEUE)

				except Full:
					#print("\t\t\t WaitingQueue is full (size = ", self._queueForWaitingVideos.qsize(),
					#      "); stuff VideoReader back to Data list.")
					with self._lockForDataList:
						self._listOfData.insert(0, videoReader)

		else:
			time.sleep(0.001)
