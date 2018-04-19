#!/usr/bin/python3

import sys
import settings.EvaluationSettings as evalSettings
from src.Classifier import Classifier
from src.Evaluator import *
import time

def PrintHelp():
	print("Usage:  Evaluate.py  $(PATH_TO_DATA_SET_CATELOG)  $(Threshold)")
	print("    or, Evaluate.py  $(PATH_TO_DATA_SET_CATELOG)")
	print("        to find best threshold.")

def PrintResults(loss_, isThresholdOptimized_, threshold_, accuracy_, duration_):
	floatPrecision = "{0:.8f}"
	if isThresholdOptimized_:
		print("\t\t loss: ", "{0:.8f}".format(loss_),
			",\t best frame threshold: ", threshold_,
			",\t accuracy: ", "{0:.8f}".format(accuracy_),
			",\t duration: ", "{0:.4f}".format(duration_), "(s)\n" )
	else:
		print("\t\t loss: ", "{0:.8f}".format(loss_),
			",\t frame threshold: ", threshold_,
			",\t\t accuracy: ", "{0:.8f}".format(accuracy_),
			",\t duration: ", "{0:.4f}".format(duration_), "(s)\n" )

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
				print("Start evaluate: ", PATH_TO_DATA_SET_CATELOG, ", and find the best threshold.")
				loss, threshold, accuracy = evaluator.Evaluate( session,
										currentEpoch_=0,
										threshold_=None)
				endEvaluateTime = time.time()
				PrintResults(loss_=loss, isThresholdOptimized_=True,
					     threshold_=threshold, accuracy_=accuracy,
					     duration_=(endEvaluateTime-startEvaluateTime) )

			else:
				threshold = int(sys.argv[2])
				print("Start evaluate: ", PATH_TO_DATA_SET_CATELOG, ", with threshold : ", threshold)
				loss, threshold, accuracy = evaluator.Evaluate( session,
										currentEpoch_=0,
										threshold_=threshold)
				endEvaluateTime = time.time()
				PrintResults(loss_=loss, isThresholdOptimized_=False,
					     threshold_=threshold, accuracy_=accuracy,
					     duration_=(endEvaluateTime-startEvaluateTime) )


		
		evaluator.Release()

	else:
		PrintHelp()
