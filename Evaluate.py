#!/usr/bin/python3

import sys
import settings.EvaluationSettings as evalSettings
from src.Classifier import Classifier
from src.Evaluator import *
import time

def PrintHelp():
	print("Usage:  Evaluate.py  $(PATH_TO_DATA_SET_CATELOG)  $(Threshold)")
	print("    or, Evaluate.py  $(PATH_TO_DATA_SET_CATELOG)")
	print("        to find the best threshold.")

def PrintResults(loss_, frameAccuracy_, isThresholdOptimized_, threshold_, videoAccuracy_, duration_):
	floatPrecision = "{0:.4f}"
	if isThresholdOptimized_:
			print("\t     loss:", floatPrecision.format(loss_),
				"     frame accuracy:", floatPrecision.format(frameAccuracy_),
				"     best frame threshold:", threshold_,
				"     video accuracy:", floatPrecision.format(videoAccuracy_),
				"     duration:", "{0:.2f}".format(duration_) + "(s)\n" )
	else:
			print("\t     loss:", floatPrecision.format(loss_),
				"     frame accuracy:", floatPrecision.format(frameAccuracy_),
				"     given frame threshold:", threshold_,
				"     video accuracy:", floatPrecision.format(videoAccuracy_),
				"     duration:", "{0:.2f}".format(duration_) + "(s)\n" )


if __name__ == '__main__':
	numberOfArguments = len(sys.argv)
	if (numberOfArguments ==2)or(numberOfArguments==3):
		PATH_TO_DATA_SET_CATELOG = sys.argv[1]
		classifier = Classifier()
		classifier.Build()
		evaluator = Evaluator("evaluate", PATH_TO_DATA_SET_CATELOG, classifier)

		with tf.Session() as session:
			init = tf.global_variables_initializer()
			session.run(init)

			print("Load Model from: ", evalSettings.PATH_TO_MODEL_CHECKPOINTS)
			modelLoader = tf.train.Saver()
			modelLoader.restore(session, evalSettings.PATH_TO_MODEL_CHECKPOINTS)

			startEvaluateTime = time.time()
			if numberOfArguments == 2:
				print("Start evaluate: ", PATH_TO_DATA_SET_CATELOG, ", and find the best threshold...")
				loss, frameAccuracy, threshold, videoAccuracy = evaluator.Evaluate(	session,
												currentEpoch_=0,
												threshold_=None)
				endEvaluateTime = time.time()
				PrintResults(loss_=loss, frameAccuracy_=frameAccuracy, isThresholdOptimized_=True,
					     threshold_=threshold, videoAccuracy_=videoAccuracy,
					     duration_=(endEvaluateTime-startEvaluateTime) )

			else:
				threshold = int(sys.argv[2])
				print("Start evaluate: ", PATH_TO_DATA_SET_CATELOG, ", with threshold : ", threshold)
				loss, frameAccuracy, threshold, videoAccuracy = evaluator.Evaluate(	session,
												currentEpoch_=0,
												threshold_=threshold)
				endEvaluateTime = time.time()
				PrintResults(loss_=loss, frameAccuracy_=frameAccuracy, isThresholdOptimized_=False,
					     threshold_=threshold, videoAccuracy_=videoAccuracy,
					     duration_=(endEvaluateTime-startEvaluateTime) )


		
		evaluator.Release()

	else:
		PrintHelp()
