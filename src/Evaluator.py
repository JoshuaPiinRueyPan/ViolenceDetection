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
		self._predictionsOp = classifier_.predictionsOp
		self._accuracyCalculator = VideosAccuracyCalculator()

		self._summaryWriter = tf.summary.FileWriter( os.path.join(trainSettings.PATH_TO_SAVE_MODEL, EVALUATOR_TYPE_) )

		# Pause DataManager to save MainMemory
		self._dataManager.Pause()

		self._bestThreshold = -1

	def __del__(self):
		print("Stop EvaluationDataManager")
		self._dataManager.Stop()


	def SetMergedSummaryOp(self, allSummariesOp_):
		self._summaryOp = allSummariesOp_

	def SetGraph(self, graph_):
		self._summaryWriter.add_graph(graph_)

	def Evaluate(self, tf_session_, currentEpoch_, threshold_=None):
		self._dataManager.Continue()
		self._listOfPreviousCellState = None

		totalLosses = 0.0
		countOfLosses = 0
		while True:
			currentLoss = self._calculateValidationForSingleBatch(tf_session_)

			countOfLosses += currentLoss.size
			totalLosses += np.sum(currentLoss)

			if self._dataManager.isNewVideo:
				self._listOfPreviousCellState = None

			if self._dataManager.isAllDataTraversed:
				break


		self._dataManager.Pause()
		meanLoss = totalLosses / countOfLosses

		if threshold_ == None:
			threshold, accuracy = self._accuracyCalculator.CalculateBestAccuracyAndThreshold(self._summaryWriter,
													 currentEpoch_)
		else:
			accuracy, _, _ = self._accuracyCalculator.CalculateAccuracyAtGivenThreshold(threshold_)
			threshold = threshold_

		self._accuracyCalculator.Reset()

		summary = tf.Summary()
		summary.value.add(tag='loss', simple_value=meanLoss)
		summary.value.add(tag='accuracy', simple_value=accuracy)
		self._summaryWriter.add_summary(summary, currentEpoch_)

		self._dataManager.Pause()

		return meanLoss, threshold, accuracy

	def Release(self):
		print("Evaluator.Release()")
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

		tupleOfOutputs = session.run( [self._lossOp, self._predictionsOp] + self._classifier.net.GetListOfStatesTensorInLSTMs(),
			     		      feed_dict = inputFeedDict )
		listOfOutputs = list(tupleOfOutputs)
		batchLoss = listOfOutputs.pop(0)
		predictions = listOfOutputs.pop(0)
		self._listOfPreviousCellState = listOfOutputs

		self._accuracyCalculator.AppendNetPredictions(predictions, batchData.batchOfLabels)

		return batchLoss

