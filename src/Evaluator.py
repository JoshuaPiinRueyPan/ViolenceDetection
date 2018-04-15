from src.data.DataManager import EvaluationDataManager
from src.AccuracyCalculator import *
import os

class Evaluator:
	def __init__(self, EVALUATOR_TYPE_, PATH_TO_DATA_CATELOG_, crossEntropyLossOp_, predictionsOp_):
		self.dataManager = EvaluationDataManager(PATH_TO_DATA_CATELOG_)
		self.lossOp = crossEntropyLossOp_
		self.predictionsOp = predictionsOp_
		self.accuracyCalculator = VideosAccuracyCalculator()

		self.validaSumWriter = tf.summary.FileWriter( os.join(trainSettings.PATH_TO_SAVE_MODEL, EVALUATOR_TYPE_) )

		# Pause DataManager to save MainMemory
		self.dataManager.Pause()

	def SetClassifierAndSummary(self, classifier_, allSummariesOp_):
		self.classifier = classifier_
		self.summaryOp = allSummariesOp_

	def Evaluate(self, tf_session, shouldSaveSummary_):
		self.dataManager.Continue()
		self.listOfPreviousCellState = None

		totalLoss = 0
		totalCorrectCount = 0
		while not self.dataManager.isAllDataTraversed:
			currentLoss, currentCorrectCount = self._calculateValidationForSingleBatch(tf_session, shouldSaveSummary_)

'''
	TODO:
		tf.reduce_mean()?  the totalLoss should be an array, not value
'''
	
			totalLoss += currentLoss
			totalCorrectCount += currentCorrectCount

			if self.dataManager.isNewVideo:
				self.listOfPreviousCellState = None

			if self.dataManager.isAllDataTraversed:
				self.dataManager.Pause()

				if shouldSaveSummary_:
					summary = tf.Summary()
					summary.value.add(tag='loss', simple_value=meanLoss)
					summary.value.add(tag='accuracy', simple_value=meanAccu)
					self.validaSumWriter.add_summary(summary, self.dataManager.epoch)
				print("    validation:")
				print("        loss: " + str(meanLoss) + ", accuracy: " + str(meanAccu) + "\n")


			valLoss += net.CalculateLoss(valDataSet.GetBatchOfData())
			if valDataSet.isNewVideo:
				net.ResetCellState()

			if valDataSet.isAllDataTraversed:
				valDataSet.Pause()

		

	def _calculateValidationForSingleBatch(self, session, shouldSaveSummary):
		batchData = BatchData()
		self.dataManager.AssignBatchData(batchData)
		
		inputFeedDict = { self.inputImage : batchData.batchOfImages,
				  self.BATCH_SIZE : batchData.batchSize,
				  self.UNROLLED_SIZE : batchData.unrolledSize,
				  self.isTraining : False,
				  self.trainingStep : 0,
				  self.groundTruth : batchData.batchOfLabels }
		cellStateFeedDict = self.classifier.GetFeedDictOfLSTM(self.listOfPreviousCellState)

		tupleOfOutputs = session.run( [self.lossOp, self.predictionsOp] + self.net.GetListOfStatesInLSTM(),
			     		      feed_dict = inputFeedDict.update(cellStateFeedDict) )
		listOfOutputs = list(tupleOfOutputs)
		loss = listOfOutputs.pop(0)
		predictions = listOfOutputs.pop(0)
		self.listOfPreviousCellState = listOfOutputs


	     



		if shouldSaveSummary:
			summary = tf.Summary()
			summary.ParseFromString(summaryValue)
			summary.value.add(tag='loss', simple_value=lossValue)
			summary.value.add(tag='accuracy', simple_value=accuValue)
			self.validaSumWriter.add_summary(summary, self.dataManager.epoch)
		print("    validation:")
		print("        loss: " + str(lossValue) + ", accuracy: " + str(accuValue) + "\n")

		return loss

	def calculateValidationOneByOne(self, session, shouldSaveSummary):
		'''
		When deal with a Large Model, stuff all validation set into a batch is not possible.
		Therefore, following stuff each validation data at a time
		'''
		arrayOfValidaLoss = np.zeros( (trainSettings.NUMBER_OF_VALIDATION_DATA) )
		arrayOfValidaAccu = np.zeros( (trainSettings.NUMBER_OF_VALIDATION_DATA) )
		for i in range(trainSettings.NUMBER_OF_VALIDATION_DATA):
			validaImage = self.validation_x[i]
			validaImage = np.reshape(validaImage,
						 [1, RadarImage.DATA_WIDTH, RadarImage.DATA_HEIGHT, RadarImage.DATA_CHANNELS])

			validaAngle = self.validation_x_angle[i]
			validaAngle = np.reshape(validaAngle, [1, 1])

			validaLabel = self.validation_y[i]
			validaLabel = np.reshape(validaLabel, [1, 2])
			lossValue, accuValue  =  session.run( [ self.lossOp, self.accuracyOp],
							      feed_dict={	self.net.isTraining : False,
										self.net.trainingStep : self.dataManager.step,
										self.net.inputImage : validaImage,
										self.net.inputAngle : validaAngle,
										self.net.groundTruth : validaLabel})
			arrayOfValidaLoss[i] = lossValue
			arrayOfValidaAccu[i] = accuValue

		meanLoss = np.mean(arrayOfValidaLoss)
		meanAccu = np.mean(arrayOfValidaAccu)

		if shouldSaveSummary:
			summary = tf.Summary()
			summary.value.add(tag='loss', simple_value=meanLoss)
			summary.value.add(tag='accuracy', simple_value=meanAccu)
			self.validaSumWriter.add_summary(summary, self.dataManager.epoch)
		print("    validation:")
		print("        loss: " + str(meanLoss) + ", accuracy: " + str(meanAccu) + "\n")


