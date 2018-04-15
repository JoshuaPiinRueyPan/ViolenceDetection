import TrainSettings as trainSettings
from src.data.DataManager import TrainDataManager

class Trainer:
	def __init__(self, classifier_):
		self._dataManager = TrainDataManager()
		self._learningRatePlaceHolder = tf.placeholder(tf.float32, shape=[])
		self._crossEntropyLossOp = classifier_.crossEntropyLossOp
		self._updateNetOp = classifier_.updateOp
		self._predictionsOp = classifier_.predictionsOp
		self._accuracyCalculator = VideosAccuracyCalculator()

		try:
			# If there's other losses (e.g. Regularization Loss)
			otherLossOp = tf.losses.get_total_loss(add_regularization_losses=True)
			totalLossOp = self.crossEntropyLoss + otherLossOp
		except:
			# If there's no other loss op
			totalLossOp = self.crossEntropyLoss

		optimizer = trainSettings.GetOptimizer(self._learningRatePlaceHolder)
		gradients = optimizer.compute_gradients(totalLossOp)
		self._drawGradients(gradients)
		self._optimzeOp = optimizer.apply_gradients(gradients)

		self.summaryWriter = tf.summary.FileWriter(trainSettings.PATH_TO_SAVE_MODEL+"/train")

	def SetMergedSummaryOp(self, allSummariesOp_):
		self._summaryOp = allSummariesOp_

	@property
	def currentEpoch(self):
		return self._dataManager.epoch

	@property
	def isNewEpoch(self):
		return self._dataManager.isNewEpoch

	def PauseDataLoading(self):
		self._dataManager.Pause()

	def ContinueDataLoading(self):
		self._dataManager.Continue()

	def PrepareNewBatchData(self):
		self._batchData = BatchData()
		self._dataManager.AssignBatchData(self._batchData)

	def Train(self, tf_session_, ):
		_backPropergateNet(tf_session_)
		_updateNet(tf_session_)


	def _backPropergateNet(self, session_):
		currentLearningRate = trainSettings.GetLearningRate(self._dataManager.epoch)

		inputFeedDict = { self.inputImage : self._batchData.batchOfImages,
				  self.BATCH_SIZE : self._batchData.batchSize,
				  self.UNROLLED_SIZE : self._batchData.unrolledSize,
				  self.isTraining : True,
				  self.trainingStep : 0,
				  self.groundTruth : self._batchData.batchOfLabels }
		'''
		    For Training, do not use previous state.  Set the argument:
		    'listOfPreviousStateValues_'=None to ensure using the zeros
		    as LSTM state.
		'''
		cellStateFeedDict = self.classifier.GetFeedDictOfLSTM(self._batchData.batchSize, listOfPreviousStateValues_=None)

		session.run( [self._optimzeOp],
		 	     feed_dict = inputFeedDict.update(cellStateFeedDict) )


	def _updateNet(self, session_, batchData_):
		'''
		    Some Network has variables that need to be updated after training (e.g. the net with
		    batch normalization).  After training, following code update such variables.
		'''
		inputFeedDict = { self.inputImage : self._batchData.batchOfImages,
				  self.BATCH_SIZE : self._batchData.batchSize,
				  self.UNROLLED_SIZE : self._batchData.unrolledSize,
				  self.isTraining : False,
				  self.trainingStep : 0,
				  self.groundTruth : self._batchData.batchOfLabels }
		cellStateFeedDict = self.classifier.GetFeedDictOfLSTM(self._batchData.batchSize, listOfPreviousStateValues_=None)

		session.run( [self._updateNetOp],
		 	     feed_dict = inputFeedDict.update(cellStateFeedDict) )



	def EvaluateTrainLoss(self, session_, threshold=None):
		'''
		    Evaluate training loss, accuracy.
		    Note: This function does not back propergate or change net weightings.
		'''
		inputFeedDict = { self.inputImage : self._batchData.batchOfImages,
				  self.BATCH_SIZE : self._batchData.batchSize,
				  self.UNROLLED_SIZE : self._batchData.unrolledSize,
				  self.isTraining : False,
				  self.trainingStep : 0,
				  self.groundTruth : self._batchData.batchOfLabels }
		cellStateFeedDict = self.classifier.GetFeedDictOfLSTM(self._batchData.batchSize, listOfPreviousStateValues_=None)

		batchLoss, predictions, _ = session.run( [self._lossOp, self._predictionsOp, self._summaryOp],
			     				 feed_dict = inputFeedDict.update(cellStateFeedDict) )

		self._accuracyCalculator.AppendNetPredictions(predictions, self._batchData.arrayOfLabels)

		if threshold_ == None:
			threshold, accuracy = self._accuracyCalculator.CalculateBestAccuracyAndThreshold(self.summaryWriter,
													 currentEpoch_)
		else:
			threshold = threshold_
			accuracy = self._accuracyCalculator.CalculateAccuracyAtGivenThreshold(threshold_)


		summary = tf.Summary()
		summary.ParseFromString(summaryValue)
		summary.value.add(tag='loss', simple_value=batchLoss)
		summary.value.add(tag='accuracy', simple_value=accuracy)

		self.summaryWriter.add_summary(summary, self._dataManager.epoch)

		return batchLoss, threshold, accuracy


	def _drawGradients(self, gradientsInfo_):
		for eachGradient, eachVariable in gradientsInfo_:
			if eachGradient is not None:
				tf.summary.histogram(eachVariable.op.name + '/gradient', eachGradient)

