#!/usr/bin/python3

from src.Classifier import Classifier
from src.Trainer import Trainer

class Main:
	def __init__(self):
		classifier = Classifier()
		classifier.Build()

		self.trainer = Trainer(classifier)
		self.validationEvaluator = Evaluator("validation", dataSettings.PATH_TO_VAL_SET_CATELOG, classifier)
		self.testEvaluator = Evaluator(dataSettings.PATH_TO_TEST_SET_CATELOG, classifier)

		summaryOp = tf.summary.merge_all()

		self.trainer.SetMergedSummaryOp(summaryOp)
		self.validationEvaluator.SetMergedSummaryOp(summaryOp)
		self.bestThreshold = None
		self.testEvaluator.SetMergedSummaryOp(summaryOp)

		self.modelSaver = tf.train.Saver(max_to_keep=trainSettings.MAX_TRAINING_SAVE_MODEL)

		self.session = tf.Session()
		init = tf.global_variables_initializer()
		self.session.run(init)

	def Run(self):
		recoverFromPretrainModelIfRequired()

		self.calculateValidationBeforeTraining()

		while self.trainer.currentEpoch < trainSettings.MAX_TRAINING_EPOCH:
			self.trainer.PrepareNewBatchData()
			self.trainer.Train()

			if self.trainer.isNewEpoch:
				print("Epoch: " + str(self.trainer.currentEpoch)+" ======================================")
				self.trainer.PauseDataLoading()

				self.calculateTrainingSetAndPrint(self.trainer.currentEpoch)
				self.calculateValidationSetAndPrint(self.trainer.currentEpoch)
				self.calculateTestSetAndPrint(self.trainer.currentEpoch)

				self.trainer.ContinueDataLoading()

				if self.trainer.currentEpoch >= trainSettings.EPOCHS_TO_START_SAVE_MODEL:
					self.saveCheckpoint(modelSaver, self.trainer.currentEpoch)
		print("Optimization finished!")


	def recoverFromPretrainModelIfRequired():
		if trainSettings.PRETRAIN_MODEL_PATH_NAME != "":
			print("Load Pretrain model from: " + trainSettings.PRETRAIN_MODEL_PATH_NAME)
			listOfAllVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			variablesToBeRecovered = [ eachVariable for eachVariable in listOfAllVariables \
						   if eachVariable.name.split('/')[0] not in \
						   trainSettings.NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT ]
			modelLoader = tf.train.Saver(variablesToBeRecovered)
			modelLoader.restore(self.session, trainSettings.PRETRAIN_MODEL_PATH_NAME)

	def calculateTrainingSetAndPrint():
		meanLoss, threshold, accuracy = trainer.EvaluateTrainLoss(self.session, self.threshold)
		print("\t train:")
		if self.threshold == None:
			print("\t\t loss: ", meanLoss. ",\t optimized frame threshold: ", threshold, ",\t accuracy: ", accuracy)
		else:
			print("\t\t loss: ", meanLoss. ",\t frame threshold: ", threshold, ",\t accuracy: ", accuracy)

	def calculateValidationBeforeTraining():
		print("Validation before Training  ======================================")
		if trainSettings.PRETRAIN_MODEL_PATH_NAME != "":
			self.calculateValidationAndPrint(currentEpoch_=0)

	def calculateValidationSetAndPrint(currentEpoch_):
		meanLoss, threshold, accuracy = self.validationEvaluator.Evaluate(self.session,
										  currentEpoch_=currentEpoch_,
										  threshold_=None)
		self.bestThreshold = threshold
		print("\t validation")
		print("\t\t loss: ", meanLoss. ",\t optimized frame threshold: ", threshold, ",\t accuracy: ", accuracy)


	def calculateTestSetAndPrint(currentEpoch_):
		if (currentEpoch_ % 10) == 0:
			meanLoss, threshold, accuracy = testEvaluator.Evaluate(self.session,
									       currentEpoch_=currentEpoch_,
									       threshold_=None)
			print("\t test")
			print("\t\t loss: ", meanLoss. ",\t optimized frame threshold: ", threshold, ",\t accuracy: ", accuracy)

		else:
			meanLoss, _, accuracy = testEvaluator.Evaluate(	self.session,
									currentEpoch_=currentEpoch_,
									threshold_=self.bestThreshold)
			print("\t test")
			print("\t\t loss: ", meanLoss. ",\t frame threshold: ", threshold, ",\t accuracy: ", accuracy)

	def saveCheckpoint(currentEpoch_):
		pathToSaveCheckpoint = os.path.join(trainSettings.PATH_TO_SAVE_MODEL, "save_epoch_" + str(currentEpoch_) )
		checkpointPathFileName = os.path.join(pathToSaveCheckpoint, "ViolenceNet.ckpt")
		self.modelSaver.save(self.session, checkpointPathFileName)



if __name__ == "__main__":
	main = Main()
	main.Run()
