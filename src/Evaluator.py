from src.data.DataManager import EvaluationDataManager, BatchData
from src.AccuracyCalculator import *
import settings.TrainSettings as trainSettings
import os

class Evaluator:
	def __init__(self, EVALUATOR_TYPE_, PATH_TO_DATA_CATELOG_, classifier_):
		'''
		    EVALUATOR_TYPE_ should be 'validation' or 'test'
		'''
		self._dataManager = EvaluationDataManager(PATH_TO_DATA_CATELOG_)
		self._classifier = classifier_
		self._lossOp = classifier_.crossEntropyLossOp
		self._correctPredictionsOp = classifier_.correctPredictionsOp
		self._predictionsOp = classifier_.predictionsOp
		self._accuracyCalculator = VideosAccuracyCalculator()

		self._summaryWriter = tf.summary.FileWriter( os.path.join(trainSettings.PATH_TO_SAVE_MODEL, EVALUATOR_TYPE_) )

		# Pause DataManager to save MainMemory
		self._dataManager.Pause()

		self._bestThreshold = -1

	def __del__(self):
		self._dataManager.Stop()


	def SetMergedSummaryOp(self, allSummariesOp_):
		self._summaryOp = allSummariesOp_

	def SetGraph(self, graph_):
		self._summaryWriter.add_graph(graph_)

	def Evaluate(self, tf_session_, currentEpoch_, threshold_=None):
		self._dataManager.Continue()
		self._listOfPreviousCellState = None

		totalLosses = 0.0
		totalCorrectPredictions = 0.0
		countOfData = 0
		while True:
			currentLoss, correctPredictions = self._calculateValidationForSingleBatch(tf_session_)

			if currentLoss.size != correctPredictions.size:
				errorMessage = " batchLoss.shape = " + str(currentLoss.shape) + "\n"
				errorMessage += " Not equal to correctPredictions.shape = " + str(correctPredictions.shape)
				raise ValueError(errorMessage)

			countOfData += currentLoss.size
			totalLosses += np.sum(currentLoss)
			totalCorrectPredictions += np.sum(correctPredictions)

			if self._dataManager.isNewVideo:
				self._listOfPreviousCellState = None

			if self._dataManager.isAllDataTraversed:
				break



		self._dataManager.Pause()
		meanLoss = totalLosses / countOfData
		frameAccuracy = totalCorrectPredictions / countOfData

		if threshold_ == None:
			threshold, videoAccuracy = self._accuracyCalculator.CalculateBestAccuracyAndThreshold(self._summaryWriter,
													 currentEpoch_)
		else:
			videoAccuracy, _, _ = self._accuracyCalculator.CalculateAccuracyAtGivenThreshold(threshold_)
			threshold = threshold_

		self._accuracyCalculator.Reset()

		summary = tf.Summary()
		summary.value.add(tag='Loss', simple_value=meanLoss)
		summary.value.add(tag='FrameAccuracy', simple_value=frameAccuracy)
		summary.value.add(tag='VideoAccuracy', simple_value=videoAccuracy)
		self._summaryWriter.add_summary(summary, currentEpoch_)


		return meanLoss, frameAccuracy, threshold, videoAccuracy

	def Release(self):
		self._dataManager.Stop()
		

	def _calculateValidationForSingleBatch(self, session):
		batchData = BatchData()
		self._dataManager.AssignBatchData(batchData)
		
		inputFeedDict = { self._classifier.inputImage : batchData.batchOfImages,
				  self._classifier.batchSize : batchData.batchSize,
				  self._classifier.unrolledSize : batchData.unrolledSize,
				  self._classifier.isTraining : False,
				  self._classifier.trainingStep : 0,
				  self._classifier.groundTruth : batchData.batchOfLabels }
		cellStateFeedDict = self._classifier.net.GetFeedDictOfLSTM(batchData.batchSize, self._listOfPreviousCellState)

		inputFeedDict.update(cellStateFeedDict)

		tupleOfOutputs = session.run( [	self._lossOp, self._correctPredictionsOp, self._predictionsOp]	\
						+ self._classifier.net.GetListOfStatesTensorInLSTMs(),
			     		      feed_dict = inputFeedDict )
		listOfOutputs = list(tupleOfOutputs)
		batchLoss = listOfOutputs.pop(0)
		correctPredictions = listOfOutputs.pop(0)
		predictions = listOfOutputs.pop(0)
		self._listOfPreviousCellState = listOfOutputs

		self._accuracyCalculator.AppendNetPredictions(predictions, batchData.batchOfLabels)

		return batchLoss, correctPredictions

