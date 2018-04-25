import tensorflow as tf		
import numpy as np
import settings.DeploySettings as deploySettings
import settings.DataSettings as dataSettings
import settings.NetSettings as netSettings

class OutputSmoother:
	def __init__(self):
		self._previousPrediction = False
		self._previousOutput = False
		self._countOfNeighborResult = 0

	def Smooth(self, isFighting_):
		if isFighting_ != self._previousPrediction:
			self._countOfNeighborResult = 1
			self._previousPrediction = isFighting_

		elif isFighting_ == self._previousPrediction:
			self._countOfNeighborResult += 1
			if self._countOfNeighborResult >= deploySettings.CHANGE_JUDGEMENT_THRESHOLD:
				self._previousOutput = isFighting_
			

		return self._previousOutput


class ViolenceDetector:
	def __init__(self):
		# PlaceHolders
		self._inputPlaceholder = tf.placeholder(dtype=dataSettings.FLOAT_TYPE,
							shape=[	1, 1, dataSettings.GROUPED_SIZE,
								dataSettings.IMAGE_SIZE,
								dataSettings.IMAGE_SIZE,
								dataSettings.IMAGE_CHANNELS] )
		self._batchSizePlaceholder = tf.placeholder(tf.int32)
		self._unrolledSizePlaceholder = tf.placeholder(tf.int32)
		self._isTrainingPlaceholder = tf.placeholder(tf.bool)
		self._trainingStepPlaceholder = tf.placeholder(tf.int64)

		# Previous Frames Holder
		self._listOfPreviousFrames = []
		self._groupedInput = None

		# Net
		self._net = netSettings.GetNetwork(	self._inputPlaceholder,
							self._batchSizePlaceholder,
							self._unrolledSizePlaceholder,
							self._isTrainingPlaceholder,
							self._trainingStepPlaceholder)
		self._net.Build()
		self._predictionOp = tf.nn.softmax(self._net.logitsOp, axis=-1, name="tf.nn.softmax")
		self._listOfPreviousCellState = None

		# Session
		self.session = tf.Session()
		init = tf.global_variables_initializer()
		self.session.run(init)
		self._recoverModelFromCheckpoints()

		# Output
		self._unsmoothedResults = []
		self._outputSmoother = OutputSmoother()

	def __del__(self):
		self.session.close()

	def Detect(self, netInputImage_):
		'''
		      The argument 'netInputImage_' should be shape of:
		    [dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_CHANNELS]
		    And the value of each pixel should be in the range of [-1, 1].
		      Note, if you use OpenCV to read images or videos, you should convert the Color from
		    BGR to RGB.  Moreover, the value should also be converted from [0, 255] to [-1, 1].
		'''
		if dataSettings.GROUPED_SIZE == 1:
			self._groupedInput = netInputImage_.reshape(self._inputPlaceholder.shape)

		else:
			self._updateGroupedInputImages(netInputImage_)

		inputFeedDict = { self._inputPlaceholder : self._groupedInput,
				  self._batchSizePlaceholder : 1,
				  self._unrolledSizePlaceholder : 1,
				  self._isTrainingPlaceholder : False,
				  self._trainingStepPlaceholder : 0 }
		cellStateFeedDict = self._net.GetFeedDictOfLSTM(1, self._listOfPreviousCellState)

		inputFeedDict.update(cellStateFeedDict)

		tupleOfOutputs = self.session.run( [self._predictionOp] + self._net.GetListOfStatesTensorInLSTMs(),
			     			   feed_dict = inputFeedDict )
		listOfOutputs = list(tupleOfOutputs)
		prediction = listOfOutputs.pop(0)
		self._listOfPreviousCellState = listOfOutputs

		isFighting = np.equal(np.argmax(prediction), np.argmax(dataSettings.FIGHT_LABEL))
		self._unsmoothedResults.append(isFighting)

		smoothedOutput = self._outputSmoother.Smooth(isFighting)

		return smoothedOutput

	@property
	def unsmoothedResults(self):
		return self._unsmoothedResults

	def _updateGroupedInputImages(self, newInputImage_):
		if len(self._listOfPreviousFrames) == dataSettings.GROUPED_SIZE:
			# Abandon the unsed frame
			self._listOfPreviousFrames.pop(0)
			self._listOfPreviousFrames.append(newInputImage_)

		else:
			blackFrame = np.full( shape=[dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_CHANNELS],
					      fill_value=-1.0,
					      dtype=dataSettings.FLOAT_TYPE)
			for i in range(dataSettings.GROUPED_SIZE-1):
				self._listOfPreviousFrames.append(blackFrame)

			self._listOfPreviousFrames.append(newInputImage_)


		self._groupedInput = np.concatenate(self._listOfPreviousFrames)
		self._groupedInput = self._groupedInput.reshape(self._inputPlaceholder.shape)
			
			

	def _recoverModelFromCheckpoints(self):
		print("Load Model from: ", deploySettings.PATH_TO_MODEL_CHECKPOINTS)
		modelLoader = tf.train.Saver()
		modelLoader.restore(self.session, deploySettings.PATH_TO_MODEL_CHECKPOINTS)


