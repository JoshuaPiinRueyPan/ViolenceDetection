import tensorflow as tf
import settings.LayerSettings as layerSettings
import settings.NetSettings as netSettings
import settings.DataSettings as dataSettings
import settings.TrainSettings as trainSettings

class Classifier:
	def __init__(self):
		self.inputImage = tf.placeholder(dataSettings.FLOAT_TYPE,
						 [None, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_CHANNELS])
		self.BATCH_SIZE = tf.placeholder(tf.int64)
		self.UNROLLED_SIZE = tf.placeholder(tf.int64)
		self.isTraining = tf.placeholder(tf.bool)
		self.trainingStep = tf.placeholder(tf.int64)
		self.groundTruth = tf.placeholder(dataSettings.FLOAT_TYPE, [None, outSettings.NUMBER_OF_CATEGORIES])

		self.net = netSettings.GetNetwork(self.inputImage, self.BATCH_SIZE, self.UNROLLED_SIZE,
						  self.isTraining, self.trainingStep)

	def Build(self):
		'''
		    Note: The return value:
			  crossEntropy.shape: [batchSize, unrolledSize]
			  predictions.shape: [batchSize, unrolledSize, NUMBER_OF_CATEGORIES]
		'''
		self.net.Build()
		self._predictions = tf.nn.softmax(self.net.logits, axis=-1, name="tf.nn.softmax")
		with tf.name_scope("Loss"):
			self._crossEntropyOp = tf.nn.softmax_cross_entropy_with_logits(	logits=self.net.logits,
											labels=self.groundTruth,
											dim=-1,
											name="tf.nn.softmax_cross_entropy_with_logits")

	@property
	def predictionsOp(self):
		return self._predictions

	@property
	def crossEntropyLossOp(self):
		return self._crossEntropyOp

	@property
	def updateOp(self):
		return self._net.updateOp


