from src.data.DataManager import EvaluationDataManager
from src.AccuracyCalculator import *
import os

class Evaluator:
	def __init__(self, EVALUATOR_TYPE_, PATH_TO_DATA_CATELOG_, classifier_):
		'''
		    EVALUATOR_TYPE_ should be 'validation' or 'test'
		'''
		self._dataManager = EvaluationDataManager(PATH_TO_DATA_CATELOG_)
		self.classifier = classifier_
		self._lossOp = classifier_.crossEntropyLossOp
		self._predictionsOp = classifier_.predictionsOp
		self._accuracyCalculator = VideosAccuracyCalculator()

		self._sumWriter = tf.summary.FileWriter( os.join(trainSettings.PATH_TO_SAVE_MODEL, EVALUATOR_TYPE_) )

		# Pause DataManager to save MainMemory
		self._dataManager.Pause()

		self._bestThreshold = -1


	def SetMergedSummaryOp(self, allSummariesOp_):
		self.summaryOp = allSummariesOp_

	def Evaluate(self, tf_session_, currentEpoch_, threshold_=None):
		self._dataManager.Continue()
		self.listOfPreviousCellState = None

		totalLoss = 0
		videoCounted = 0
		totalCorrectCount = 0
		while not self._dataManager.isAllDataTraversed:
			currentLoss = self._calculateValidationForSingleBatch(tf_session_)

			totalLoss += currentLoss
			videoCount += 1

			if self._dataManager.isNewVideo:
				self.listOfPreviousCellState = None


		self._dataManager.Pause()
		meanLoss = totalLoss / videoCounted
		if threshold_ == None:
			threshold, accuracy = self._accuracyCalculator.CalculateBestAccuracyAndThreshold(self.summaryWriter,
													 currentEpoch_)
		else:
			accuracy = self._accuracyCalculator.CalculateAccuracyAtGivenThreshold(threshold_)
			threshold = threshold_

		self._accuracyCalculator.Reset()

		summary = tf.Summary()
		summary.value.add(tag='loss', simple_value=meanLoss)
		summary.value.add(tag='accuracy', simple_value=accuracy)
		self._sumWriter.add_summary(summary, currentEpoch_)

		self.dataManager.Pause()

		return meanLoss, threshold, accuracy


	def _calculateValidationForSingleBatch(self, session, shouldSaveSummary):
		batchData = BatchData()
		self._dataManager.AssignBatchData(batchData)
		
		inputFeedDict = { self.inputImage : batchData.batchOfImages,
				  self.BATCH_SIZE : batchData.batchSize,
				  self.UNROLLED_SIZE : batchData.unrolledSize,
				  self.isTraining : False,
				  self.trainingStep : 0,
				  self.groundTruth : batchData.batchOfLabels }
		cellStateFeedDict = self.classifier.GetFeedDictOfLSTM(batchData.batchSize, self.listOfPreviousCellState)

		tupleOfOutputs = session.run( [self._lossOp, self._predictionsOp] + self.net.GetListOfStatesTensorInLSTMs(),
			     		      feed_dict = inputFeedDict.update(cellStateFeedDict) )
		listOfOutputs = list(tupleOfOutputs)
		batchLoss = listOfOutputs.pop(0)
		predictions = listOfOutputs.pop(0)
		self.listOfPreviousCellState = listOfOutputs

		self._accuracyCalculator.AppendNetPredictions(predictions, batchData.arrayOfLabels)

		return batchLoss

