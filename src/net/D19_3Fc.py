import tensorflow as tf
from src.net.NetBase import *
from src.layers.BasicLayers import *
from src.layers.RNN import *
import settings.LayerSettings as layerSettings
import settings.DataSettings as dataSettings
import settings.TrainSettings as trainSettings
import numpy as np

DARKNET19_MODEL_PATH = 'data/pretrainModels/darknet19/darknet19.pb'

class Net(NetworkBase):
	def __init__(self, inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_):
		self._inputImage = inputImage_
		self._batchSize = batchSize_
		self._unrolledSize = unrolledSize_
		self._isTraining = isTraining_
		self._trainingStep = trainingStep_
	
		self._DROPOUT_VALUE = 0.5

		self._dictOfInterestedActivations = {}

		if trainSettings.UNROLLED_SIZE != 1:
			errorMessage = __name__ + " only take UNROLLED_SIZE = 1 (single frame inference);\n"
			errorMessage += "However, TrainSettings.UNROLLED_SIZE = " + str(trainSettings.UNROLLED_SIZE)
			raise ValueError(errorMessage)

		if dataSettings.GROUPED_SIZE != 1:
			errorMessage = __name__ + " only take GROUPED_SIZE = 1;\n"
			errorMessage += "However, DataSettings.GROUPED_SIZE = " + str(dataSettings.GROUPED_SIZE)
			raise ValueError(errorMessage)

	def Build(self):
		darknet19_GraphDef = tf.GraphDef()

		'''
		      The CNN only take input shape [..., w, h, c].  Thus, move the UNROLLED_SIZE dimension
		    to merged with BATCH_SIZE, and form the shape: [b*u, w, h, c].
		'''
		convInput = tf.reshape(self._inputImage, [-1,
							  dataSettings.IMAGE_SIZE, dataSettings.IMAGE_SIZE, dataSettings.IMAGE_CHANNELS])

		with tf.name_scope("DarkNet19"):
			with open(DARKNET19_MODEL_PATH, 'rb') as modelFile:
				darknet19_GraphDef.ParseFromString(modelFile.read())
				listOfOperations = tf.import_graph_def(darknet19_GraphDef,
									input_map={"input": convInput},
									return_elements=["BiasAdd_13"])
#									return_elements=["32-leaky"])
#									return_elements=["BiasAdd_14"])
#									return_elements=["34-leaky"])
#									return_elements=["BiasAdd_15"])
#									return_elements=["36-leaky"])
#									return_elements=["BiasAdd_16"])
#									return_elements=["38-leaky"])
#									return_elements=["BiasAdd_17"])
#									return_elements=["40-leaky"])
#									return_elements=["Pad_18"])
#									return_elements=["41-convolutional"])
#									return_elements=["BiasAdd_18"])
				lastOp = listOfOperations[-1]
				darknetOutput = lastOp.outputs[0]
			
		print("darknetOutput.shape = ", darknetOutput.shape)

		out = FullyConnectedLayer('Fc1', darknetOutput, numberOfOutputs_=4096)
		out, updateOp1 = BatchNormalization('BN1', out, isConvLayer_=False,
						     isTraining_=self._isTraining, currentStep_=self._trainingStep)

		out = FullyConnectedLayer('Fc2', out, numberOfOutputs_=4096)
		out, updateOp2 = BatchNormalization('BN2', out, isConvLayer_=False,
						     isTraining_=self._isTraining, currentStep_=self._trainingStep)
		

		out = FullyConnectedLayer('Fc3', out, numberOfOutputs_=dataSettings.NUMBER_OF_CATEGORIES)
		self._logits = tf.reshape(out, [self._batchSize, self._unrolledSize, -1])

		self._updateOp = tf.group(updateOp1, updateOp2)

	@property
	def logitsOp(self):
		return self._logits

	@property
	def updateOp(self):
		return self._updateOp


	def GetListOfStatesTensorInLSTMs(self):
		# No states should be update
		return []


	def GetFeedDictOfLSTM(self, BATCH_SIZE_, listOfPreviousStateValues_=None):
		# No additional feed_dict
		return {}
