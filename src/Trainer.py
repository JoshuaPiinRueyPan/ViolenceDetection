import tensorflow as tf
import settings.TrainSettings as trainSettings
import settings.DataSettings as dataSettings
from src.data.DataManager import TrainDataManager, BatchData
from src.AccuracyCalculator import *

class Trainer:
	def __init__(self, classifier_):
		self._dataManager = TrainDataManager(dataSettings.PATH_TO_TRAIN_SET_CATELOG)
		self._learningRatePlaceHolder = tf.placeholder(tf.float32, shape=[])
		self._classifier = classifier_
		self._accuracyCalculator = VideosAccuracyCalculator()

		try:
			# If there's other losses (e.g. Regularization Loss)
			otherLossOp = tf.losses.get_total_loss(add_regularization_losses=True)
			totalLossOp = self._classifier.crossEntropyLossOp + otherLossOp
		except:
			# If there's no other loss op
			totalLossOp = self._classifier.crossEntropyLossOp

		optimizer = trainSettings.GetOptimizer(self._learningRatePlaceHolder)
		gradients = optimizer.compute_gradients(totalLossOp)
		self._drawGradients(gradients)
		self._optimzeOp = optimizer.apply_gradients(gradients)

		self._summaryWriter = tf.summary.FileWriter(trainSettings.PATH_TO_SAVE_MODEL+"/train")

	def __del__(self):
		print("Stop TrainDataManager")
		self._dataManager.Stop()

	def SetMergedSummaryOp(self, allSummariesOp_):
		self._summaryOp = allSummariesOp_

	def SetGraph(self, graph_):
		self._summaryWriter.add_graph(graph_)

	@property
	def currentEpoch(self):
		return self._dataManager.epoch

	@property
	def isNewEpoch(self):
		return self._dataManager.isNewEpoch

	@property
	def dataLoaderInfo(self):
		return self._dataManager.GetQueueInfo()

	def PauseDataLoading(self):
		self._dataManager.Pause()

	def ContinueDataLoading(self):
		self._dataManager.Continue()

	def PrepareNewBatchData(self):
		self._batchData = BatchData()
		self._dataManager.AssignBatchData(self._batchData)

	def Train(self, tf_session_):
		self._backPropergateNet(tf_session_)
		self._updateNet(tf_session_)

	def Release(self):
		print("Trainer.Release()")
		self._dataManager.Stop()

	def _backPropergateNet(self, session_):
		currentLearningRate = trainSettings.GetLearningRate(self._dataManager.epoch, self._dataManager.step)

		inputFeedDict = { self._classifier.inputImage : self._batchData.batchOfImages,
				  self._classifier.batchSize : self._batchData.batchSize,
				  self._classifier.unrolledSize : self._batchData.unrolledSize,
				  self._classifier.isTraining : True,
				  self._classifier.trainingStep : self._dataManager.step,
				  self._classifier.groundTruth : self._batchData.batchOfLabels,
				  self._learningRatePlaceHolder : currentLearningRate }

		'''
		    For Training, do not use previous state.  Set the argument:
		    'listOfPreviousStateValues_'=None to ensure using the zeros
		    as LSTM state.
		'''
		cellStateFeedDict = self._classifier.net.GetFeedDictOfLSTM(self._batchData.batchSize, listOfPreviousStateValues_=None)

		inputFeedDict.update(cellStateFeedDict)

		session_.run( [self._optimzeOp],
		 	      feed_dict = inputFeedDict )

		if self._dataManager.isNewEpoch:
			summary = tf.Summary()
			summary.value.add(tag='LearningRate', simple_value=currentLearningRate)
			self._summaryWriter.add_summary(summary, self._dataManager.epoch)



	def _updateNet(self, session_):
		'''
		    Some Network has variables that need to be updated after training (e.g. the net with
		    batch normalization).  After training, following code update such variables.
		'''
		inputFeedDict = { self._classifier.inputImage : self._batchData.batchOfImages,
				  self._classifier.batchSize : self._batchData.batchSize,
				  self._classifier.unrolledSize : self._batchData.unrolledSize,
				  self._classifier.isTraining : False,
				  self._classifier.trainingStep : self._dataManager.step,
				  self._classifier.groundTruth : self._batchData.batchOfLabels }
		cellStateFeedDict = self._classifier.net.GetFeedDictOfLSTM(self._batchData.batchSize, listOfPreviousStateValues_=None)

		inputFeedDict.update(cellStateFeedDict)

		session_.run( [self._classifier.updateOp],
		 	     feed_dict = inputFeedDict )



	def EvaluateTrainLoss(self, session_, threshold_=None):
		'''
		    Evaluate training loss, accuracy.
		    Note: This function does not back propergate or change net weightings.
		'''
		inputFeedDict = { self._classifier.inputImage : self._batchData.batchOfImages,
				  self._classifier.batchSize : self._batchData.batchSize,
				  self._classifier.unrolledSize : self._batchData.unrolledSize,
				  self._classifier.isTraining : False,
				  self._classifier.trainingStep : self._dataManager.step,
				  self._classifier.groundTruth : self._batchData.batchOfLabels }
		cellStateFeedDict = self._classifier.net.GetFeedDictOfLSTM(self._batchData.batchSize, listOfPreviousStateValues_=None)

		inputFeedDict.update(cellStateFeedDict)

		batchLoss, predictions, summaryValue = session_.run( [self._classifier.crossEntropyLossOp,
								      self._classifier.predictionsOp,
								      self._summaryOp],
			     					     feed_dict = inputFeedDict )
		meanLoss = np.mean(batchLoss)

		self._accuracyCalculator.AppendNetPredictions(predictions, self._batchData.batchOfLabels)

		if threshold_ == None:
			threshold, accuracy = self._accuracyCalculator.CalculateBestAccuracyAndThreshold(self._summaryWriter,
													 self._dataManager.epoch)
		else:
			threshold = threshold_
			accuracy, _, _ = self._accuracyCalculator.CalculateAccuracyAtGivenThreshold(threshold_)


		summary = tf.Summary()
		summary.ParseFromString(summaryValue)
		summary.value.add(tag='loss', simple_value=meanLoss)
		summary.value.add(tag='accuracy', simple_value=accuracy)

		self._summaryWriter.add_summary(summary, self._dataManager.epoch)

		return meanLoss, threshold, accuracy


	def _drawGradients(self, gradientsInfo_):
		for eachGradient, eachVariable in gradientsInfo_:
			if eachGradient is not None:
				tf.summary.histogram(eachVariable.op.name + '/gradient', eachGradient)

