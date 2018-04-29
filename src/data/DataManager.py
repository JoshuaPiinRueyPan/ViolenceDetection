from abc import ABCMeta, abstractmethod
import random
import numpy as np
import time
import threading
import settings.TrainSettings as trainSettings
import settings.EvaluationSettings as evalSettings
import settings.DataSettings as dataSettings
from src.data.VideoData import VideoData
import src.data.DataAugmenter as DataAugmenter

import six
if six.PY2:
	from Queue import *
else:
	from queue import *

'''
   Number of Data to Produce when One data is Consumed.
'''
PRODUCE_CONSUME_RATIO = 15

IS_DEBUG_MODE = False

class BatchData:
	def __init__(self):
		self.batchSize = 0
		self.unrolledSize = 0
		self.groupedSize = 0
		self.batchOfImages = None
		self.batchOfLabels = None

class DataManagerBase:
	__metaclass__ = ABCMeta
	def __init__(self, PATH_TO_DATA_SET_CATELOG_, WAITING_QUEUE_MAX_SIZE_, LOADED_QUEUE_MAX_SIZE_):
		self._listOfData = []
		self._initVideoData(PATH_TO_DATA_SET_CATELOG_)
		self._lockForThreadControl = threading.Lock()
		self._shouldPause = False
		self._shouldStop = False
		self._listOfLoadDataThreads = []

		self._lockForDataList = threading.Lock()
		self._queueForWaitingVideos = Queue(maxsize=WAITING_QUEUE_MAX_SIZE_)
		self._queueForLoadedVideos = Queue(maxsize=LOADED_QUEUE_MAX_SIZE_)

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
			print("Send Stop singal to Loading threads...")
			print("\t The Loading threads will Stop in about " + str(dataSettings.TIMEOUT_FOR_WAIT_QUEUE) + " (s).")

	def GetQueueInfo(self):
		info = "listOfData.len() = " + str( len(self._listOfData) ) + ";\t"
		info += "WaitingQueue.len() = " + str( self._queueForWaitingVideos.qsize() ) + ";\t"
		info += "LoadedQueue.len() = " + str(self._queueForLoadedVideos.qsize() ) + ";\t"
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

				except TimeoutError:
					time.sleep(0.1)

				
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

				try:
					self._queueForWaitingVideos.put(videoReader, block=False)

				except Full:
					with self._lockForDataList:
						self._listOfData = [videoReader] + self._listOfData
			except IndexError:
				'''
				    For the case that WAITING_QUEUE_MAX_SIZE > TOTAL_DATA,
				    the IndexError may be raised from '_listOfData.pop()'.
				'''
				if IS_DEBUG_MODE:
					print("\t\t\t ** In DataManager:")
					print("\t\t\t\t All data in self._listOfData is pushed to the WaitingQueue or LoadedQueue.")
					print("\t\t\t\t You may want to reduce the WAITING_QUEUE_MAX_SIZE...")
				else:
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

	def putLoadedVideosBackToWaitingQueueOrDataListIfQueueIsFull(self, listOfLoadedVideos_):
		try:
			while len(listOfLoadedVideos_) > 0:
				eachVideoReader = listOfLoadedVideos_.pop(0)
				eachVideoReader.ReleaseImages()
				self._queueForWaitingVideos.put(eachVideoReader, block=True,
								timeout=dataSettings.TIMEOUT_FOR_WAIT_QUEUE)
		except Full:
			if IS_DEBUG_MODE:
				print("\t\t\t\t WaitingQueue is full (size =", self._queueForWaitingVideos.qsize(),
				      "); put VideoReader back to data list")
			listOfLoadedVideos_.insert(0, eachVideoReader)
			with self._lockForDataList:
				self._listOfData = listOfLoadedVideos_ + self._listOfData



	@abstractmethod
	def AssignBatchData(self):
		pass

	@abstractmethod
	def _loadVideoData(self):
		pass


class TrainDataManager(DataManagerBase):
	def __init__(self, PATH_TO_DATA_SET_CATELOG_):
		super().__init__(PATH_TO_DATA_SET_CATELOG_,
				 WAITING_QUEUE_MAX_SIZE_=trainSettings.WAITING_QUEUE_MAX_SIZE,
				 LOADED_QUEUE_MAX_SIZE_=trainSettings.LOADED_QUEUE_MAX_SIZE)

		# Public variables
		self._epoch = 0
		self._step = 0

		self._isNewEpoch = True
		self._dataCursor = 0

		if len(self._listOfData) < (trainSettings.NUMBER_OF_LOAD_DATA_THREADS*trainSettings.BATCH_SIZE):
			errorMessage = "NUMBER_OF_TRAIN_DATA(=" + str(len(self._listOfData)) + ")"
			errorMessage += " < trainSettings.NUMBER_OF_LOAD_DATA_THREADS * BatchSize(=" 
			errorMessage += str(trainSettings.NUMBER_OF_LOAD_DATA_THREADS*trainSettings.BATCH_SIZE) + ")!\n"
			errorMessage += "This will cause DeadLock, since each loading thread can't get "
			errorMessage += "all batch data.\n"
			errorMessage += "Reduce the trainSettings.NUMBER_OF_LOAD_DATA_THREADS, or get More data!"
			raise ValueError(errorMessage)

		if trainSettings.BATCH_SIZE > (trainSettings.WAITING_QUEUE_MAX_SIZE):
			errorMessage += "BATCH_SIZE(=" + str(trainSettings.BATCH_SIZE) + ")"
			errorMessage += " > TrainSettings.WAITING_QUEUE_MAX_SIZE)\n"
			errorMessage += "This will cause DeadLock, since each loading thread can't get "
			errorMessage += "all batch data.\n"
			errorMessage += "Reduce the trainSettings.NUMBER_OF_LOAD_DATA_THREADS, or Increate the Queue size"
			raise ValueError(errorMessage)
			

		self.pushVideoDataToWaitingQueue(trainSettings.WAITING_QUEUE_MAX_SIZE)

		self._executeLoadDataThreads(trainSettings.NUMBER_OF_LOAD_DATA_THREADS)

	def AssignBatchData(self, batchData_):
		'''
		      The user should pass BatchData as argument to this function,
		    since this would be faster then this function return two numpy.array.

		      The 'batchData_.batchOfImages' will be assigned as the shape:
		    [BATCH_SIZE, UNROLLED_SIZE, GROUPED_SIZE, w, h, c].
		      The 'batchData_.batchOfLabels' will be assigned as the shape:
		    [BATCH_SIZE, UNROLLED_SIZE, 2].
		'''
		self._isNewEpoch = False

		try:
			batchData = self._queueForLoadedVideos.get(block=True,
								   timeout=dataSettings.TIMEOUT_FOR_WAIT_QUEUE)

		except Empty:
			errorMessage = "In TrainDataManager:"
			errorMessage += "\t Unable to get batch data in duration: "
			errorMessage += str(dataSettings.TIMEOUT_FOR_WAIT_QUEUE) + "(s)\n"
			errorMessage += "\t TrainQueue info:\n"
			errorMessage += self.GetQueueInfo()
			raise TimeoutError(errorMessage)

		batchData_.batchSize = batchData.batchSize
		batchData_.unrolledSize = batchData.unrolledSize
		batchData_.groupedSize = batchData.groupedSize
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
		if self._queueForLoadedVideos.qsize() <= trainSettings.LOADED_QUEUE_MAX_SIZE:
			listOfLoadedVideos = []
			for i in range(trainSettings.BATCH_SIZE):
				try:
					videoReader = self._queueForWaitingVideos.get(block=True,
										      timeout=dataSettings.TIMEOUT_FOR_WAIT_QUEUE)

					if trainSettings.PERFORM_DATA_AUGMENTATION:
						videoReader.LoadVideoImages(dataAugmentFunction_=DataAugmenter.Augment)
					else:
						videoReader.LoadVideoImages()

					listOfLoadedVideos.append(videoReader)

				except Empty:
					if IS_DEBUG_MODE:
						print("\t\t\t ** In TrainDataManager:")
						print("\t\t\t\t    WaitingQueue is Empty, not enough data to form a batch.")
						print("\t\t\t\t    " + self.GetQueueInfo())
						print("\t\t\t\t    Release batch data and push VideoReader back to WaitingQueue...")
						print("\t\t\t\t    Note: You may want to reduce the thread size.")

					'''
					    Push more Data to WaitingQueue, so that other threads can keep pop from it.
					    While release the current thread videoReader and push back to WaitingQueue
					    or DataList.
					'''
					self.pushVideoDataToWaitingQueue(trainSettings.BATCH_SIZE * PRODUCE_CONSUME_RATIO)
					self.putLoadedVideosBackToWaitingQueueOrDataListIfQueueIsFull(listOfLoadedVideos)

					raise TimeoutError


			batchData = BatchData()
			batchData.batchSize = trainSettings.BATCH_SIZE
			batchData.unrolledSize = trainSettings.UNROLLED_SIZE
			batchData.groupedSize = dataSettings.GROUPED_SIZE

			arrayOfBatchImages = np.zeros( [batchData.batchSize, batchData.unrolledSize, batchData.groupedSize,
							dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_CHANNELS],
							dtype=dataSettings.FLOAT_TYPE )
			arrayOfBatchLabels = np.zeros( [batchData.batchSize, batchData.unrolledSize, 2],
							dtype=dataSettings.FLOAT_TYPE )

			for b in range(batchData.batchSize):
				currentVideo = listOfLoadedVideos[b]
				frameStartIndex = random.randint(0, 
								max(0, currentVideo.totalFrames - batchData.unrolledSize) )

				try:
					arrayOfImages, arrayOfLabels = self._getDataFromSingleVideo(currentVideo,
												    frameStartIndex,
												    batchData.unrolledSize)
				except Exception as error:
					self.putLoadedVideosBackToWaitingQueueOrDataListIfQueueIsFull(listOfLoadedVideos)
					print(error)
					print("\nException catched.  Put loadedVideos back to WaitingQueue and pass...")
					print("error occur at: b = ", b, ", currentVideo.images = ", currentVideo.images)
					raise TimeoutError

				# Release the video frames
				currentVideo.ReleaseImages()

				'''
				    To stuff the grouped input:
				    suppose VideoFrames = [a, b, c, d, e, f, g]
				    G=2: result = [ (0, a), (a, b), (b, c), (c, d), (d, e), (e, f), (f, g) ]
				    G=3: result = [ (0, 0, a), (0, a, b), (a, b, c), ..., (e, f, g) ]

				'''
				maxFrameIndex = batchData.unrolledSize - 1
				maxGroupIndex = batchData.groupedSize - 1
				for u in range(batchData.unrolledSize):
					for g in range(batchData.groupedSize):
						if (u + g - maxGroupIndex) >=0:
							arrayOfBatchImages[b, u, g] = arrayOfImages[u + g - maxGroupIndex]
						else:

							'''
							    The BLACK image is '-1.0' in the Network input, since we have transform
							    the [0, 255] to [-1., 1.].  Therefore, for the artificial frame, If the
							    BLACK color is desired, then following use np.full(-1), NOT np.zeros().
							'''
							arrayOfBatchImages[b, u, g] = np.full( shape=[dataSettings.IMAGE_SIZE,
												      dataSettings.IMAGE_SIZE,
												      dataSettings.IMAGE_CHANNELS],
											       fill_value=-1.0,
											       dtype=dataSettings.FLOAT_TYPE)

				arrayOfBatchLabels[b] = arrayOfLabels

			batchData.batchOfImages = arrayOfBatchImages
			batchData.batchOfLabels = arrayOfBatchLabels

			try:
				self._queueForLoadedVideos.put(batchData, block=True, timeout=dataSettings.TIMEOUT_FOR_WAIT_QUEUE)
				self.appendVideoDataBackToDataList(listOfLoadedVideos)

			except Full:
				if IS_DEBUG_MODE:
					print("\t\t\t LoadedQueue is full (size =", self._queueForLoadedVideos.qsize(),
					      ");  put VideoReader back to WaitingQueue (size = ",
					      self._queueForWaitingVideos.qsize(), ")")

				self.putLoadedVideosBackToWaitingQueueOrDataListIfQueueIsFull(listOfLoadedVideos)
		
		else:
			time.sleep(0.1)



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
			while True:
				valLoss += net.CalculateLoss(valDataSet.AssignBatchData())
				if valDataSet.isNewVideo:
					net.ResetCellState()

				if valDataSet.isAllDataTraversed:
					valDataSet.Pause()
					break
	'''
	def __init__(self, PATH_TO_DATA_SET_CATELOG_):
		super().__init__(PATH_TO_DATA_SET_CATELOG_,
				 WAITING_QUEUE_MAX_SIZE_=evalSettings.WAITING_QUEUE_MAX_SIZE,
				 LOADED_QUEUE_MAX_SIZE_=evalSettings.LOADED_QUEUE_MAX_SIZE)
		self._isAllDataTraversed = False
		self._isNewVideo = True
		self._dataCursor = 0
		self._currentVideo = None
		self._frameCursor = 0

		self.pushVideoDataToWaitingQueue(evalSettings.WAITING_QUEUE_MAX_SIZE)

		self._executeLoadDataThreads(evalSettings.NUMBER_OF_LOAD_DATA_THREADS)

	def AssignBatchData(self, batchData_):
		'''
		      The user should pass BatchData as argument to this function,
		    since this would be faster then this function return two numpy.array.

		      The 'batchData_.batchOfImages' will be assigned as the shape:
		    [BATCH_SIZE, UNROLLED_SIZE, GROUPED_SIZE, w, h, c].
		      The 'batchData_.batchOfLabels' will be assigned as the shape:
		    [BATCH_SIZE, UNROLLED_SIZE, 2].
		'''
		self._isAllDataTraversed = False
		self._isNewVideo = False
		if self._currentVideo == None:
			try:
				self._currentVideo = self._queueForLoadedVideos.get(block=True,
										    timeout=dataSettings.TIMEOUT_FOR_WAIT_QUEUE)

			except Empty:
				errorMessage = "In EvaluationDataManager:"
				errorMessage += "\t Unable to get batch data in duration: "
				errorMessage += dataSettings.TIMEOUT_FOR_WAIT_QUEUE + "(s)\n"
				errorMessage += "\t TrainQueue info:\n"
				errorMessage += self.GetQueueInfo()
				raise TimeoutError(errorMessage)

		unrolledSize = min(evalSettings.UNROLLED_SIZE,
				   self._currentVideo.totalFrames - self._frameCursor)

		batchData_.batchSize = 1
		batchData_.unrolledSize = unrolledSize
		batchData_.groupedSize = dataSettings.GROUPED_SIZE
		tempImages, batchData_.batchOfLabels = self._getDataFromSingleVideo(self._currentVideo,
										    self._frameCursor, unrolledSize)
		batchData_.batchOfImages = np.zeros([batchData_.batchSize, batchData_.unrolledSize, batchData_.groupedSize,
						     dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_CHANNELS],
						    dtype=dataSettings.FLOAT_TYPE)

		for b in range(batchData_.batchSize):
			maxFrameIndex = batchData_.unrolledSize - 1
			maxGroupIndex = batchData_.groupedSize - 1
			for u in range(batchData_.unrolledSize):
				for g in range(batchData_.groupedSize):
					if (u + g - maxGroupIndex) >=0:
						batchData_.batchOfImages[b, u, g] = tempImages[u + g - maxGroupIndex]
					else:
						'''
						    The BLACK image is '-1.0' in the Network input, since we have transform
						    the [0, 255] to [-1., 1.].  Therefore, for the artificial frame, If the
						    BLACK color is desired, then following use np.full(-1), NOT np.zeros().
						'''
						batchData_.batchOfImages[b, u, g] = np.full( shape=[dataSettings.IMAGE_SIZE,
												    dataSettings.IMAGE_SIZE,
												    dataSettings.IMAGE_CHANNELS],
											     fill_value=-1.0,
											     dtype=dataSettings.FLOAT_TYPE)

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
		if self._queueForLoadedVideos.qsize() <= evalSettings.LOADED_QUEUE_MAX_SIZE:
			try:
				videoReader = self._queueForWaitingVideos.get(block=False)

			except Empty:
				raise TimeoutError()

			videoReader.LoadVideoImages()

			try:
				self._queueForLoadedVideos.put(videoReader, block=True,
								timeout=dataSettings.TIMEOUT_FOR_WAIT_QUEUE)

			except:
				if IS_DEBUG_MODE:
					print("\t\t\t LoadedQueue is full (size = ", self._queueForLoadedVideos.qsize(),
					      "); stuff VideoReader back to WaitingQueue (size = ",
					      self._queueForWaitingVideos.qsize(), ")")

				self.putLoadedVideosBackToWaitingQueueOrDataListIfQueueIsFull( [videoReader] )

		else:
			time.sleep(0.1)
