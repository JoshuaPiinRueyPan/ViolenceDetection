#!/usr/bin/python3

from src.Classifier import Classifier
from src.Trainer import Trainer


def recoverFromPretrainModelIfRequired(self, session):

	if trainSettings.PRETRAIN_MODEL_PATH_NAME != "":
		print("Load Pretrain model from: " + trainSettings.PRETRAIN_MODEL_PATH_NAME)
		listOfAllVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		variablesToBeRecovered = [ eachVariable for eachVariable in listOfAllVariables \
					   if eachVariable.name.split('/')[0] not in \
					   trainSettings.NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT ]
		modelLoader = tf.train.Saver(variablesToBeRecovered)
		modelLoader.restore(session, trainSettings.PRETRAIN_MODEL_PATH_NAME)

if __name__ == "__main__":
	classifier = Classifier()
	lossOp, predictionsOp, updateOp = classifier.Build()

	trainer = Trainer(lossOp, correctCountOp)
	validationEvaluator = Evaluator(dataSettings.PATH_TO_VAL_SET_CATELOG, lossOp, correctCountOp)
	testEvaluator = Evaluator(dataSettings.PATH_TO_TEST_SET_CATELOG, lossOp, correctCountOp)

	summaryOp = tf.summary.merge_all()

	trainer.SetClassifierAndSummary(classifier, summaryOp)
	validationEvaluator.SetClassifierAndSummary(classifier, summaryOp)
	testEvaluator.SetClassifierAndSummary(classifier, summaryOp)

	saver = tf.train.Saver(max_to_keep=trainSettings.MAX_TRAINING_SAVE_MODEL)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		recoverFromPretrainModelIfRequired(sess)

		# Calculate Validation before Training
		print("Validation before Training  ======================================")
		self.CalculateValidation(sess, shouldSaveSummary=False)

		while self.dataManager.epoch < trainSettings.MAX_TRAINING_EPOCH:
			batch_x, batch_x_angle, batch_y = self.dataManager.GetTrainingBatch(trainSettings.BATCH_SIZE)
			self.trainIceNet(sess, batch_x, batch_x_angle, batch_y)
			self.updateIceNet(sess, batch_x, batch_x_angle, batch_y)

			if self.dataManager.isNewEpoch:
				print("Epoch: " + str(self.dataManager.epoch)+" ======================================")
				self.CalculateTrainingLoss(sess, batch_x, batch_x_angle, batch_y)
				self.CalculateValidation(sess, shouldSaveSummary=True)

				if self.dataManager.epoch >= trainSettings.EPOCHS_TO_START_SAVE_MODEL:
					self.saveCheckpoint(sess)
		print("Optimization finished!")

class Trainer:
	def __init__(self):


	def Run(self):




	def trainIceNet(self, session, batch_x, batch_x_angle, batch_y):
		currentLearningRate = trainSettings.GetLearningRate(self.dataManager.epoch)
		session.run( self.optimzeOp,
			     feed_dict={self.net.isTraining : True,
					self.net.trainingStep : self.dataManager.step,
					self.net.inputImage : batch_x,
					self.net.inputAngle : batch_x_angle,
					self.net.groundTruth : batch_y,
					self.learningRate : currentLearningRate})

	def updateIceNet(self, session, batch_x, batch_x_angle, batch_y):
		'''
		    Some Network has variables that need to be updated after training (e.g. the net with
		    batch normalization).  After training, following code update such variables.
		'''
		session.run( self.updateNetOp,
			     feed_dict={self.net.isTraining : False,
					self.net.trainingStep : self.dataManager.step,
					self.net.inputImage : batch_x,
					self.net.inputAngle : batch_x_angle,
					self.net.groundTruth : batch_y})


	def CalculateTrainingLoss(self, session, batch_x, batch_x_angle, batch_y):
		summaryValue, lossValue, accuValue  =  session.run( [self.summaryOp, self.lossOp, self.accuracyOp],
								    feed_dict={	self.net.isTraining : False,
										self.net.trainingStep : self.dataManager.step,
										self.net.inputImage : batch_x,
										self.net.inputAngle : batch_x_angle,
										self.net.groundTruth : batch_y})

		summary = tf.Summary()
		summary.ParseFromString(summaryValue)
		summary.value.add(tag='loss', simple_value=lossValue)
		summary.value.add(tag='accuracy', simple_value=accuValue)

		self.trainSumWriter.add_summary(summary, self.dataManager.epoch)
		print("    train:")
		print("        loss: " + str(lossValue) + ", accuracy: " + str(accuValue) + "\n")



	def saveCheckpoint(self, tf_session):
		pathToSaveCheckpoint = os.path.join(trainSettings.PATH_TO_SAVE_MODEL, "save_epoch_" + str(self.dataManager.epoch) )
		checkpointPathFileName = os.path.join(pathToSaveCheckpoint, "IceNet.ckpt")
		self.saver.save(tf_session, checkpointPathFileName)



if __name__ == "__main__":
	solver = Solver()
	solver.Run()
