import TrainSettings as trainSettings
from src.data.DataManager import TrainDataManager

class Trainer:
	def __init__(self, crossEntropyLossOp_, correctCountOp_):
		self.dataManager = TrainDataManager()
		self.learningRatePlaceHolder = tf.placeholder(tf.float32, shape=[])
		self.crossEntropyLossOp = crossEntropyLossOp_
		self.correctCountOp = correctCountOp_

		try:
			# If there's other losses (e.g. Regularization Loss)
			otherLossOp = tf.losses.get_total_loss(add_regularization_losses=True)
			totalLossOp = self.crossEntropyLoss + otherLossOp
		except:
			# If there's no other loss op
			totalLossOp = self.crossEntropyLoss

		optimizer = trainSettings.GetOptimizer(self.learningRatePlaceHolder)
		gradients = optimizer.compute_gradients(totalLossOp)
		self._drawGradients(gradients)
		self.optimzeOp = optimizer.apply_gradients(gradients)

		self.trainSumWriter = tf.summary.FileWriter(trainSettings.PATH_TO_SAVE_MODEL+"/train")

	def SetClassifierAndSummary(self, classifier_, allSummariesOp_):
		self.classifier = classifier_
		self.summaryOp = allSummariesOp_

	def _drawGradients(self, gradientsInfo_):
		for eachGradient, eachVariable in gradientsInfo_:
			if eachGradient is not None:
				tf.summary.histogram(eachVariable.op.name + '/gradient', eachGradient)

