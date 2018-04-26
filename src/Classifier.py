import tensorflow as tf
import settings.LayerSettings as layerSettings
import settings.NetSettings as netSettings
import settings.DataSettings as dataSettings

class Classifier:
	def __init__(self):
		self.inputImage = tf.placeholder(dataSettings.FLOAT_TYPE,
						 shape=[None, None, None,
							 dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_CHANNELS])
		self.batchSize = tf.placeholder(tf.int32)
		self.unrolledSize = tf.placeholder(tf.int32)
		self.isTraining = tf.placeholder(tf.bool)
		self.trainingStep = tf.placeholder(tf.int64)
		self.groundTruth = tf.placeholder(dataSettings.FLOAT_TYPE, shape=[None, None,
										  dataSettings.NUMBER_OF_CATEGORIES])

		self.net = netSettings.GetNetwork(self.inputImage,
						  self.batchSize, self.unrolledSize,
						  self.isTraining, self.trainingStep)

	def Build(self):
		'''
		    Note: The return value:
			  crossEntropy.shape: [batchSize, unrolledSize]
			  predictions.shape: [batchSize, unrolledSize, NUMBER_OF_CATEGORIES]
		'''
		self.net.Build()
		self._predictions = tf.nn.softmax(self.net.logitsOp, axis=-1, name="tf.nn.softmax")
		with tf.name_scope("Loss"):
			self._crossEntropyOp = tf.nn.softmax_cross_entropy_with_logits(	logits=self.net.logitsOp,
											labels=self.groundTruth,
											dim=-1,
											name="tf.nn.softmax_cross_entropy_with_logits")
		with tf.name_scope("FrameAccuracy"):
			self._correctPredictions = tf.equal(	tf.argmax(self._predictions, axis=-1),
								tf.argmax(self.groundTruth, axis=-1),
								name="tf.equal")
			self._correctPredictions = tf.cast(self._correctPredictions, tf.float32)

	@property
	def predictionsOp(self):
		return self._predictions

	@property
	def crossEntropyLossOp(self):
		return self._crossEntropyOp

	@property
	def correctPredictionsOp(self):
		return self._correctPredictions

	@property
	def updateOp(self):
		return self.net.updateOp


