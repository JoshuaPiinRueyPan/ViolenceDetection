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
		self.trainEvaluator = Evaluator("train", dataSettings.PATH_TO_TRAIN_SET_CATELOG, classifier)
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
		self.trainEvaluator.SetMergedSummaryOp(summaryOp)
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
				self.trainer.PauseDataLoading()

				self.evaluateValidationSetAndPrint(self.trainer.currentEpoch)
				self.evaluateTrainingSetAndPrint(self.trainer.currentEpoch)

				if trainSettings.PERFORM_DATA_AUGMENTATION:
					# Preload TrainBatch while evaluate the TestSet
					self.trainer.ContinueDataLoading()

				self.evaluateTestSetAndPrint(self.trainer.currentEpoch)

				self.trainer.ContinueDataLoading()

				self.resetTimeMeasureVariables()

				if self.trainer.currentEpoch >= trainSettings.EPOCHS_TO_START_SAVE_MODEL:
					self.saveCheckpoint(self.trainer.currentEpoch)
		print("Optimization finished.")
		self.trainer.Release()
		self.trainEvaluator.Release()
		self.validationEvaluator.Release()
		self.testEvaluator.Release()



	def recoverFromPretrainModelIfRequired(self):
		if trainSettings.PRETRAIN_MODEL_PATH_NAME != "":
			print("Load Pretrain model from: " + trainSettings.PRETRAIN_MODEL_PATH_NAME)
			listOfAllVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			variablesToBeRecovered = [ eachVariable for eachVariable in listOfAllVariables \
						   if eachVariable.name.split('/')[0] not in \
						   trainSettings.NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT ]
			modelLoader = tf.train.Saver(variablesToBeRecovered)
			modelLoader.restore(self.session, trainSettings.PRETRAIN_MODEL_PATH_NAME)

	def evaluateTrainingSetAndPrint(self, currentEpoch_):
		'''
		    Since the BATCH_SIZE may be small (= 4 in my case), its BatchLoss or BatchAccuracy
		    may be fluctuated.  Calculate the whole Training Loss instead.
		    Note: If one want to calculate the BatchLoss ONLY, use Trainer.EvaluateTrainLoss().
		'''
		startEvaluateTime = time.time()
		loss, frameAccuracy, threshold, videoAccuracy = self.trainEvaluator.Evaluate(	self.session,
												currentEpoch_=currentEpoch_,
												threshold_=self.bestThreshold)
		endEvaluateTime = time.time()

		self.printCalculationResults(jobType_='train', loss_=loss, frameAccuracy_=frameAccuracy,
					     isThresholdOptimized_=False,
					     threshold_=threshold, videoAccuracy_=videoAccuracy,
					     duration_=(endEvaluateTime-startEvaluateTime) )


	def calculateValidationBeforeTraining(self):
		if trainSettings.PRETRAIN_MODEL_PATH_NAME != "":
			print("Validation before Training ", "============================="
					+ "======================================"
					+ "======================================")
			self.evaluateValidationSetAndPrint(currentEpoch_=0)

	def evaluateValidationSetAndPrint(self, currentEpoch_):
		startEvaluateTime = time.time()
		loss, frameAccuracy, threshold, videoAccuracy = self.validationEvaluator.Evaluate(self.session,
												  currentEpoch_=currentEpoch_,
												  threshold_=None)
		endEvaluateTime = time.time()

		self.bestThreshold = threshold
		self.printCalculationResults(jobType_='validation', loss_=loss, frameAccuracy_=frameAccuracy,
					     isThresholdOptimized_=True,
					     threshold_=threshold, videoAccuracy_=videoAccuracy,
					     duration_=(endEvaluateTime-startEvaluateTime) )

	def evaluateTestSetAndPrint(self, currentEpoch_):
		startEvaluateTime = time.time()
		loss, frameAccuracy, threshold, videoAccuracy = self.testEvaluator.Evaluate(self.session,
											    currentEpoch_=currentEpoch_,
											    threshold_=self.bestThreshold)
		endEvaluateTime = time.time()

		self.printCalculationResults(jobType_='test', loss_=loss, frameAccuracy_=frameAccuracy,
					     isThresholdOptimized_=False,
					     threshold_=threshold, videoAccuracy_=videoAccuracy,
					     duration_=(endEvaluateTime-startEvaluateTime) )

	def printTimeMeasurement(self):
		timeForTrainOneEpoch = time.time() - self._startTrainEpochTime
		print("\t Back Propergation time measurement:")
		print("\t\t duration: ", "{0:.2f}".format(timeForTrainOneEpoch), "s/epoch")
		averagedTrainTime = timeForTrainOneEpoch / self._trainCountInOneEpoch
		print("\t\t average: ", "{0:.2f}".format(averagedTrainTime), "s/batch")
		print()

	def resetTimeMeasureVariables(self):
		self._startTrainEpochTime = time.time()
		self._trainCountInOneEpoch = 0

	def printCalculationResults(self, jobType_, loss_, frameAccuracy_, isThresholdOptimized_, threshold_, videoAccuracy_, duration_):
		floatPrecision = "{0:.4f}"
		print("\t "+jobType_+":")
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


	def saveCheckpoint(self, currentEpoch_):
		pathToSaveCheckpoint = os.path.join(trainSettings.PATH_TO_SAVE_MODEL, "save_epoch_" + str(currentEpoch_) )
		checkpointPathFileName = os.path.join(pathToSaveCheckpoint, "ViolenceNet.ckpt")
		self.modelSaver.save(self.session, checkpointPathFileName)



if __name__ == "__main__":
	main = Main()
	main.Run()
