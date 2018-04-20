#!/usr/bin/python3
from src.AccuracyCalculator import *
import tensorflow as tf
import time
import numpy as np
import settings.DataSettings as dataSettings

# maxCount = 8
netPrediction_1 = np.array( [[ [0.9, 0.1], [0.7, 0.3], [0.6, 0.4], [0.4, 0.6], [0.3, 0.7],
			    [0.2, 0.8], [0.1, 0.9], [0.2, 0.8], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]  ]] )
label_1 = np.array( [ [dataSettings.FIGHT_LABEL] * netPrediction_1.shape[0] ] )

# maxCount = 4
netPrediction_2 = np.array( [[ [0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.4, 0.6], [0.3, 0.7],
			    [0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]  ]] )
label_2 = np.array( [ [dataSettings.FIGHT_LABEL] * netPrediction_2.shape[0] ] )

# maxCount = 2
netPrediction_3 = np.array( [[ [0.9, 0.1], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.8, 0.2],
			    [0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.1, 0.9]  ]] )
label_3 = np.array( [ [dataSettings.NO_FIGHT_LABEL] * netPrediction_3.shape[0] ] )

# maxCount = 6
netPrediction_4 = np.array( [[ [0.9, 0.1], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.3, 0.7],
			    [0.1, 0.9], [0.1, 0.9], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]  ]] )
label_4 = np.array( [ [dataSettings.NO_FIGHT_LABEL] * netPrediction_4.shape[0] ] )


def Check_CalculateAccuracy():
	print("Check_CalculateAccuracy()")
	accuracyCalculator = VideosAccuracyCalculator()

	# 5 TP
	numberOfTP = 5
	truePositivePredictions = np.tile(netPrediction_1, [numberOfTP, 1, 1])
	truePositiveLabels = np.tile(label_1, [numberOfTP, 1, 1])
	accuracyCalculator.AppendNetPredictions(truePositivePredictions, truePositiveLabels)

	# 2 FN
	numberOfFN = 2
	falseNegativePredictions = np.tile(netPrediction_2, [numberOfFN, 1, 1])
	falseNegativeLabels = np.tile(label_2, [numberOfFN, 1, 1])
	accuracyCalculator.AppendNetPredictions(falseNegativePredictions, falseNegativeLabels)

	# 3 TN
	numberOfTN = 3
	trueNegativePredictions = np.tile(netPrediction_3, [numberOfTN, 1, 1])
	trueNegativeLabels = np.tile(label_3, [numberOfTN, 1, 1])
	accuracyCalculator.AppendNetPredictions(trueNegativePredictions, trueNegativeLabels)

	# 3 FP
	numberOfFP = 3
	falsePositivePredictions = np.tile(netPrediction_4, [numberOfFP, 1, 1])
	falsePositiveLabels = np.tile(label_4, [numberOfFP, 1, 1])
	accuracyCalculator.AppendNetPredictions(falsePositivePredictions, falsePositiveLabels)

	answerOfAccuracy = float(numberOfTP + numberOfTN) / (numberOfTP + numberOfFN + numberOfTN + numberOfFP)
	answerOfPrecision = float(numberOfTP) / (numberOfTP + numberOfFP)
	answerOfRecall = float(numberOfTP) / (numberOfTP + numberOfFN)
	
	accuracy, precision, recall = accuracyCalculator.CalculateAccuracyAtGivenThreshold(threshold_=5)
	print("\t (accuracy, precision, recall) = (", accuracy, ", ", precision, ", ", recall, ")")

	if abs(accuracy - answerOfAccuracy) >= 1e-5:
		raise ValueError("\t Accuracy (="+str(accuracy)+"); However, answer = " + str(answerOfAccuracy))

	if abs(precision - answerOfPrecision) >= 1e-5:
		raise ValueError("\t Precision (="+str(precision)+"); However, answer = " + str(answerOfPrecision))

	if abs(recall - answerOfRecall) >= 1e-5:
		raise ValueError("\t Recall (="+str(recall)+"); However, answer = " + str(answerOfRecall))

	accuracyCalculator.Reset()
	print("\t check passed.")

def Check_CalculateBestAccuracyAndThreshold():
	print("Check_CalculateBestAccuracyAndThreshold()")
	summaryWriter = tf.summary.FileWriter("src/unit_test/accuracy")

	accuracyCalculator = VideosAccuracyCalculator()
	
	accuracyCalculator.AppendNetPredictions(netPrediction_1, label_1)
	accuracyCalculator.AppendNetPredictions(netPrediction_2, label_2)
	accuracyCalculator.AppendNetPredictions(netPrediction_3, label_3)
	accuracyCalculator.AppendNetPredictions(netPrediction_4, label_4)

	bestThreshold, bestAccuracy = accuracyCalculator.CalculateBestAccuracyAndThreshold(tf_summaryWriter_=summaryWriter, currentEpoch_=1)
	print("\t (bestThreshold, bestAccuracy) = (", bestThreshold, ", ", bestAccuracy, ")")

	answerOfThreshold = 3
	answerOfAccuracy = 0.75
	if bestThreshold != answerOfThreshold:
		raise ValueError("\t bestThreshold(="+str(bestThreshold)+"); However, answer = " + str(answerOfThreshold))

	if abs(bestAccuracy - answerOfAccuracy) > 1e-5:
		raise ValueError("\t bestAccuracy(="+str(bestAccuracy)+"); However, answer = " + str(answerOfAccuracy))

	accuracyCalculator.Reset()
	print("\t check passed.")

def Check_ProcessingTime():
	print("Check_ProcessingTime()")
	
	summaryWriter = tf.summary.FileWriter("src/unit_test/accuracy")

	accuracyCalculator = VideosAccuracyCalculator()
	
	predictionOfAllVideos = np.zeros([400, 40, 2])
	labelOfAllVideos = np.zeros([400, 40, 2])

	for i in range(400):  # Test set has 400 videos
		for j in range(40):  # Videos has 40 frames in average.
			fightProbility = np.random.rand()
			predictionOfAllVideos[i, j, :] = [1 - fightProbility,  fightProbility]

		isFightLabel = np.random.rand() >= 0.5
		if isFightLabel:
			labelOfAllVideos[i, :, :] = np.tile(dataSettings.FIGHT_LABEL, [40, 1])
		else:
			labelOfAllVideos[i, :, :] = np.tile(dataSettings.NO_FIGHT_LABEL, [40, 1])
		

	startAppendTime = time.time()
	accuracyCalculator.AppendNetPredictions(predictionOfAllVideos, labelOfAllVideos)
	endAppendTime = time.time()

	print("\t Averaged AppendTime: ", endAppendTime - startAppendTime)

	startCalculateTime = time.time()
	bestThreshold, bestAccuracy = accuracyCalculator.CalculateBestAccuracyAndThreshold(tf_summaryWriter_=summaryWriter, currentEpoch_=2)
	endCalculateTime = time.time()

	print("\t\t (bestThreshold, bestAccuracy) = (", bestThreshold, ", ", bestAccuracy, ")")

	print("\t Calculate Best Accuracy time: ", endCalculateTime - startCalculateTime)

	accuracyCalculator.Reset()
	print("\t check passed.")
			

if __name__ == "__main__":
	print()
	Check_CalculateAccuracy()
	print()

	print()
	Check_CalculateBestAccuracyAndThreshold()
	print()

	print()
	Check_ProcessingTime()
	print()
