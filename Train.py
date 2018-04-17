#!/usr/bin/python3

from src.Evaluator import *
from src.Classifier import Classifier
from src.Trainer import Trainer
import settings.DataSettings as dataSettings
import time

class Main:
	def __init__(self):
		classifier = Classifier()
		classifier.Build()

		# Trainer, Evaluator
		print("Reading Training set...")
		self.trainer = Trainer(classifier)
		print("\t Done.\n")
		print("Reading Validation set...")
		self.validationEvaluator = Evaluator("validation", dataSettings.PATH_TO_VAL_SET_CATELOG, classifier)
		print("\t Done.\n")
		print("Reading Test set...")
		self.testEvaluator = Evaluator("test", dataSettings.PATH_TO_TEST_SET_CATELOG, classifier)
		print("\t Done.\n")

		# Summary
		summaryOp = tf.summary.merge_all()
		self.trainer.SetMergedSummaryOp(summaryOp)
		self.validationEvaluator.SetMergedSummaryOp(summaryOp)
		self.bestThreshold = None
		self.testEvaluator.SetMergedSummaryOp(summaryOp)

		# Time
		self._startTrainEpochTime = time.time()
		self._trainCountInOneEpoch = 0

		# Saver
		self.modelSaver = tf.train.Saver(max_to_keep=trainSettings.MAX_TRAINING_SAVE_MODEL)

		# Session
		self.session = tf.Session()
		init = tf.global_variables_initializer()
		self.session.run(init)

		self.trainer.SetGraph(self.session.graph)
		self.validationEvaluator.SetGraph(self.session.graph)

	def __del__(self):
		self.session.close()

	def Run(self):
		self.recoverFromPretrainModelIfRequired()

		self.calculateValidationBeforeTraining()
		self.resetTimeMeasureVariables()

		print("Path to save mode: ", trainSettings.PATH_TO_SAVE_MODEL)
		print("\nStart Training...\n")

		while self.trainer.currentEpoch < trainSettings.MAX_TRAINING_EPOCH:
			self.trainer.PrepareNewBatchData()
			self.trainer.Train(self.session)
			self._trainCountInOneEpoch += 1

			if self.trainer.isNewEpoch:
				print("Epoch:", self.trainer.currentEpoch, "======================================"
					+ "======================================"
					+ "======================================")

				self.printTimeMeasurement()
				self.evaluateTrainingSetAndPrint()
				self.evaluateValidationSetAndPrint(self.trainer.currentEpoch)
				self.evaluateTestSetAndPrint(self.trainer.currentEpoch)

				self.resetTimeMeasureVariables()

				if self.trainer.currentEpoch >= trainSettings.EPOCHS_TO_START_SAVE_MODEL:
					self.saveCheckpoint(self.trainer.currentEpoch)
		print("Optimization finished.")



	def recoverFromPretrainModelIfRequired(self):
		if trainSettings.PRETRAIN_MODEL_PATH_NAME != "":
			print("Load Pretrain model from: " + trainSettings.PRETRAIN_MODEL_PATH_NAME)
			listOfAllVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			variablesToBeRecovered = [ eachVariable for eachVariable in listOfAllVariables \
						   if eachVariable.name.split('/')[0] not in \
						   trainSettings.NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT ]
			modelLoader = tf.train.Saver(variablesToBeRecovered)
			modelLoader.restore(self.session, trainSettings.PRETRAIN_MODEL_PATH_NAME)

	def evaluateTrainingSetAndPrint(self):
		startEvaluateTime = time.time()
		loss, threshold, accuracy = self.trainer.EvaluateTrainLoss(self.session, self.bestThreshold)
		endEvaluateTime = time.time()

		if self.bestThreshold == None:
			self.printCalculationResults(jobType_='train', loss_=loss, isThresholdOptimized_=True,
						     threshold_=threshold, accuracy_=accuracy,
						     duration_=(endEvaluateTime-startEvaluateTime) )
		else:
			self.printCalculationResults(jobType_='train', loss_=loss, isThresholdOptimized_=False,
						     threshold_=threshold, accuracy_=accuracy,
						     duration_=(endEvaluateTime-startEvaluateTime) )

	def calculateValidationBeforeTraining(self):
		if trainSettings.PRETRAIN_MODEL_PATH_NAME != "":
			print("Validation before Training ", "======================================"
					+ "======================================"
					+ "======================================")
			self.calculateValidationAndPrint(currentEpoch_=0)

	def evaluateValidationSetAndPrint(self, currentEpoch_):
		startEvaluateTime = time.time()
		loss, threshold, accuracy = self.validationEvaluator.Evaluate(self.session,
										  currentEpoch_=currentEpoch_,
										  threshold_=None)
		endEvaluateTime = time.time()

		self.bestThreshold = threshold
		self.printCalculationResults(jobType_='validation', loss_=loss, isThresholdOptimized_=True,
					     threshold_=threshold, accuracy_=accuracy,
					     duration_=(endEvaluateTime-startEvaluateTime) )

	def evaluateTestSetAndPrint(self, currentEpoch_):
		if (currentEpoch_!=0)and(currentEpoch_ % 3) == 0:
			startEvaluateTime = time.time()
			loss, threshold, accuracy = self.testEvaluator.Evaluate(self.session,
									   currentEpoch_=currentEpoch_,
									   threshold_=None)
			endEvaluateTime = time.time()

			self.printCalculationResults(jobType_='test', loss_=loss, isThresholdOptimized_=True,
						     threshold_=threshold, accuracy_=accuracy,
						     duration_=(endEvaluateTime-startEvaluateTime) )

		else:
			startEvaluateTime = time.time()
			loss, threshold, accuracy = self.testEvaluator.Evaluate(self.session,
								   currentEpoch_=currentEpoch_,
								   threshold_=self.bestThreshold)
			endEvaluateTime = time.time()

			self.printCalculationResults(jobType_='test', loss_=loss, isThresholdOptimized_=False,
						     threshold_=threshold, accuracy_=accuracy,
						     duration_=(endEvaluateTime-startEvaluateTime) )

	def printTimeMeasurement(self):
		timeForTrainOneEpoch = time.time() - self._startTrainEpochTime
		print("\t Back Propergation time measurement:")
		print("\t\t duration: ", "{0:.4f}".format(timeForTrainOneEpoch), "s/epoch")
		averagedTrainTime = timeForTrainOneEpoch / self._trainCountInOneEpoch
		print("\t\t average: ", "{0:.4f}".format(averagedTrainTime), "s/batch")
		print()

		queueInfo = self.trainer.dataLoaderInfo
		print("\t Training Queue info:")
		print("\t\t" + queueInfo + "\n")
	
	def resetTimeMeasureVariables(self):
		self._startTrainEpochTime = time.time()
		self._trainCountInOneEpoch = 0

	def printCalculationResults(self, jobType_, loss_, isThresholdOptimized_, threshold_, accuracy_, duration_):
		floatPrecision = "{0:.8f}"
		print("\t "+jobType_+":")
		if isThresholdOptimized_:
			print("\t\t loss: ", "{0:.8f}".format(loss_),
				",\t best frame threshold: ", threshold_,
				",\t\t accuracy: ", "{0:.8f}".format(accuracy_),
				",\t duration: ", "{0:.4f}".format(duration_), "(s)\n" )
		else:
			print("\t\t loss: ", "{0:.8f}".format(loss_),
				",\t frame threshold: ", threshold_,
				",\t accuracy: ", "{0:.8f}".format(accuracy_),
				",\t duration: ", "{0:.4f}".format(duration_), "(s)\n" )


	def saveCheckpoint(self, currentEpoch_):
		pathToSaveCheckpoint = os.path.join(trainSettings.PATH_TO_SAVE_MODEL, "save_epoch_" + str(currentEpoch_) )
		checkpointPathFileName = os.path.join(pathToSaveCheckpoint, "ViolenceNet.ckpt")
		self.modelSaver.save(self.session, checkpointPathFileName)



if __name__ == "__main__":
	main = Main()
	main.Run()
