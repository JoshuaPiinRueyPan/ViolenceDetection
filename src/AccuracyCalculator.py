import tensorflow as tf
import settings.DataSettings as dataSettings
import numpy as np

class VideosAccuracyCalculator:
	def __init__(self):
		self._listOfVideoPredictions = []

	def AppendNetPredictions(self, netPredictions_, arrayOfLabels_):
		if ( len(netPredictions_.shape)!=3 )or( netPredictions_.shape[-1] != 2 ):
			errorMessage = "netPredictions_.shape = " + str(netPredictions_.shape)
			errorMessage += "; However, the expected shape is (UNROLLED_SIZE_, 1)"
			raise ValueError(errorMessage)

		if netPredictions_.shape[0] != arrayOfLabels_.shape[0]:
			errorMessage = "netPredictions_.shape = " + str(netPredictions_.shape)
			errorMessage += "; However, arrayOfLabels_.shape = " + str(arrayOfLabels_.shape)
			errorMessage += "\n\t The two array has different BATCH_SIZE!"
			raise ValueError(errorMessage)

		for videoIndex, eachVideoPredictions in enumerate(netPredictions_):
			unrolledSize = eachVideoPredictions.shape[0]
			maxFightDurationCounted = self._countMaxFightDurationOfSingleVideo(eachVideoPredictions)
			isFightLabel = self._determineLabelIsFight(arrayOfLabels_[videoIndex])
			self._listOfVideoPredictions.append( [unrolledSize, maxFightDurationCounted, isFightLabel] )

	def CalculateAccuracyAtGivenThreshold(self, threshold_):
		TOTAL_VIDEOS = float(len(self._listOfVideoPredictions))

		countOfTP = 0  # TP: True Positive
		countOfFP = 0  # FP: False Positive
		countOfTN = 0  # TN: True Negative
		countOfFN = 0  # FN: False Negative
		for eachVideoPrediction in self._listOfVideoPredictions:
			_, maxFightDurationCounted, isFightLabel = eachVideoPrediction
			isPredictionPositive = maxFightDurationCounted >= threshold_
			if (isFightLabel)and(isPredictionPositive):  # True Positive
				countOfTP += 1

			elif (not isFightLabel)and(isPredictionPositive):  # False Positive
				countOfFP += 1

			elif (not isFightLabel)and(not isPredictionPositive):  # True Negative
				countOfTN += 1

			elif (isFightLabel)and(not isPredictionPositive):  # False Negative
				countOfFN += 1

			else:
				errorMessage = "You may forget to consider the situation: "
				errorMessage += "(isFightLabel, isPredictionPositive) = ("
				errorMessage += str(isFightLabel) + ", " + str(isPredictionPositive) + ")"
				raise ValueError(errorMessage)

		accuracy = (countOfTP + countOfTN) / (TOTAL_VIDEOS + 1e-9)
		precision = countOfTP / (countOfTP + countOfFP + 1e-9)
		recall = countOfTP / (countOfTP + countOfFN + 1e-9)

		return accuracy, precision, recall

	def CalculateBestAccuracyAndThreshold(self, tf_summaryWriter_=None, currentEpoch_=None):
		MIN_UNROLLS_OF_VIDEOS = self._getMinimumUnrollsInVideos()+1
		bestThreshold = -1
		bestAccuracy = 0.0
		for currentThreshold in range(1, MIN_UNROLLS_OF_VIDEOS):
			currentAccuracy, currentPrecision, currentRecall = self.CalculateAccuracyAtGivenThreshold(currentThreshold)

			if tf_summaryWriter_ != None:
				self._drawThresholdAccuracyCurve(tf_summaryWriter_, epoch_=currentEpoch_,
								 threshold_=currentThreshold, accuracy_=currentAccuracy)
				self._draw_PR_Curve(tf_summaryWriter_, epoch_=currentEpoch_,
						   precision_=currentPrecision, recall_=currentRecall)
						
			if currentAccuracy > bestAccuracy:
				bestThreshold = currentThreshold
				bestAccuracy = currentAccuracy


		return bestThreshold, bestAccuracy


	def Reset(self):
		del self._listOfVideoPredictions[:]

	def _determineLabelIsFight(self, labelOfSingleVideoFrames_):
		isFightCondition = labelOfSingleVideoFrames_[:, 1] >= 0.5
		fightFrameCount = np.extract(isFightCondition, labelOfSingleVideoFrames_).shape[0]
		return (fightFrameCount > 0)
	
	def _countMaxFightDurationOfSingleVideo(self, singleVideoPredictions_):
		maxFightCount = 0
		currentFightCount = 0
		for eachFramePrediction in singleVideoPredictions_:
			if np.argmax(eachFramePrediction) == np.argmax(dataSettings.FIGHT_LABEL):
				currentFightCount += 1

			else:
				# In the Ending of serial fight frames, update the maxFightCount.
				if currentFightCount > maxFightCount:
					maxFightCount = currentFightCount

				currentFightCount = 0

		if currentFightCount > maxFightCount:
			maxFightCount = currentFightCount

		return maxFightCount

	def _getMinimumUnrollsInVideos(self):
		minUnrolls = float('INF')
		for eachVideoPrediction in self._listOfVideoPredictions:
			currentUnrolls = eachVideoPrediction[0]
			if currentUnrolls < minUnrolls:
				minUnrolls = currentUnrolls

		return minUnrolls

	def _drawThresholdAccuracyCurve(self, tf_summaryWriter_, epoch_, threshold_, accuracy_):
		summary = tf.Summary()
		summary.value.add(tag='Threshold-Accuracy_Curve_epoch_'+str(epoch_), simple_value=accuracy_)
		tf_summaryWriter_.add_summary(summary, threshold_)

	def _draw_PR_Curve(self, tf_summaryWriter_, epoch_, precision_, recall_):
		'''
		    Note: Since Tensorboard can ONLY show the value of Horizontal-Axis as integer,
		          following change the float points to % representation.
		'''
		summary = tf.Summary()
		summary.value.add(tag='Precision-Recall_Curve_epoch_'+str(epoch_), simple_value=precision_*100)
		tf_summaryWriter_.add_summary(summary, recall_*100)

