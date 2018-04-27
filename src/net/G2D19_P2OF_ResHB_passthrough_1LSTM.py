import tensorflow as tf
from src.net.NetBase import *
from src.layers.LayerHelper import *
from src.layers.BasicLayers import *
from src.layers.ResidualLayers import *
from src.layers.RNN import *
import settings.LayerSettings as layerSettings
import settings.DataSettings as dataSettings
import numpy as np

DARKNET19_MODEL_PATH = 'data/pretrainModels/darknet19/darknet19.pb'

class Net(NetworkBase):
	def __init__(self, inputImage_, batchSize_, unrolledSize_, isTraining_, trainingStep_):
		self._inputImage = inputImage_
		self._batchSize = batchSize_
		self._unrolledSize = unrolledSize_
		self._isTraining = isTraining_
		self._trainingStep = trainingStep_
	
		self._DROPOUT_PROB = 0.5
		self._NUMBER_OF_NEURONS_IN_LSTM = 1024

		self._dictOfInterestedActivations = {}

		if dataSettings.GROUPED_SIZE != 2:
			errorMessage = __name__ + " only take GROUPED_SIZE = 2;\n"
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
#									return_elements=["2-maxpool", "32-leaky"])
									return_elements=["2-maxpool", "36-leaky"])
				pool2 = listOfOperations[0].outputs[0]
				lastOp = listOfOperations[-1]
				out = lastOp.outputs[0]
				out, updateOp0 = BatchNormalization('BN_0', out, isConvLayer_=True,
								     isTraining_=self._isTraining, currentStep_=self._trainingStep)
			

		opticalFlowOut, updateOF = self._buildOpticalFlowNet(pool2)
		print("opticalFlowOut.shape = ", opticalFlowOut.shape)  # shape = [b*u, 1024]

		with tf.name_scope("CNN"):
			'''
			    The input shape = [b, u, g, w, h, c]
			    after Conv, shape = [b*u*g, w', h', c']
			    here, decouple the Group dimension, shape = [b*u, g * w' * h' * c']
			'''
			print("In CNN:")
			print("\t darknetOutput.shape = ", out.shape)   # shape = [b*u*g, 7, 7, 1024]
			w, h, c = out.shape[1:]  # 7, 7, 1024
			out = tf.reshape( out,
					  [self._batchSize * self._unrolledSize, dataSettings.GROUPED_SIZE,
					  w, h, c])  # [b*u, g, 7, 7, 1024]
			out = tf.transpose(out, perm=[0, 2, 3, 4, 1])  # [b*u, 7, 7, 1024, g]
			print("\t after transpose, out.shape = ", out.shape)
			out = tf.reshape( out, 
					  [self._batchSize * self._unrolledSize,
					   w, h, c * dataSettings.GROUPED_SIZE])
			print("\t before ConcatConv, out.shape = ", out.shape)  # shape = [b*u, 7, 7, 1024*g]

			out, updateOp1 = ResidualHeadBlock('ResHeadBlock', out, [1024, 512, 1024], isTraining_=self._isTraining,
							trainingStep_=self._trainingStep, activationType_="LEAKY_RELU", isTrainable_=True)

			print("\t before Reshape, out.shape = ", out.shape)
			out = tf.reshape( out, [self._batchSize, self._unrolledSize, 7, 7, 1024] )

			self._dictOfInterestedActivations['CNN'] = out
			print("\t output.shape = ", out.shape)


		with tf.name_scope("Concat"):
			print("In Concat:")
			out = tf.concat([out, opticalFlowOut], axis=-1)
			print("\t after tf.concat(), out.shape = ", out.shape)
			out = tf.reshape( out, [self._batchSize * self._unrolledSize, 7, 7, 2048] )
			print("\t before Conv, out.shape = ", out.shape)
			self._dictOfInterestedActivations['ConcatInput'] = out
			out, updateOp2 = ResidualHeadBlock('ResHeadBlock', out, [64, 64, 256], isTraining_=self._isTraining,
							trainingStep_=self._trainingStep, activationType_="LEAKY_RELU", isTrainable_=True)
			out = tf.cond(self._isTraining, lambda: tf.nn.dropout(out, self._DROPOUT_PROB), lambda: out)
			print("\t before Fc, out.shape = ", out.shape)
			out = FullyConnectedLayer('Fc', out, numberOfOutputs_=1024)
			out, updateOp3 = BatchNormalization('BN_5', out, isConvLayer_=False,
							     isTraining_=self._isTraining, currentStep_=self._trainingStep)
			'''
			    Note: For tf.nn.rnn_cell.dynamic_rnn(), the input shape of [1:] must be explicit.
			          i.e., one Can't Reshape the out by:
				  out = tf.reshape(out, [BATCH_SIZE, UNROLLED_SIZE, -1])
				  since '-1' is implicit dimension.
			'''
			featuresShapeInOneBatch = out.shape[1:].as_list()
			targetShape = [self._batchSize, self._unrolledSize] + featuresShapeInOneBatch
			out = tf.reshape(out, targetShape)
			self._dictOfInterestedActivations['ConcatOutput'] = out
			print("before LSTM, shape = ", out.shape)


		out, self._stateTensorOfLSTM_1, self._statePlaceHolderOfLSTM_1 = LSTM(	"LSTM_1",
											out,
											self._NUMBER_OF_NEURONS_IN_LSTM,
											isTraining_=self._isTraining,
											dropoutProb_=self._DROPOUT_PROB)

		self._dictOfInterestedActivations['LSTM'] = out

		with tf.name_scope("Fc_Final"):
			featuresShapeInOneBatch = out.shape[2:].as_list()
			targetShape = [self._batchSize * self._unrolledSize] + featuresShapeInOneBatch
			out = tf.reshape(out, targetShape)
			out = FullyConnectedLayer('Fc3', out, numberOfOutputs_=dataSettings.NUMBER_OF_CATEGORIES)
			self._logits = tf.reshape(out, [self._batchSize, self._unrolledSize, -1])

		self._updateOp = tf.group(updateOF, updateOp0, updateOp1, updateOp2, updateOp3)
		print()



	@property
	def logitsOp(self):
		return self._logits

	@property
	def updateOp(self):
		return self._updateOp


	def GetListOfStatesTensorInLSTMs(self):
		'''
		    You should Not Only sess.run() the net.logits, but also this listOfTensors
		    to get the States of LSTM.  And assign it to PlaceHolder next time.
		    ex:
			>> tupleOfResults = sess.run( [out] + net.GetListOfStatesTensorInLSTMs(), ...)
			>> listOfResults = list(tupleOfResults)
			>> output = listOfResults.pop(0)
			>> listOfStates = listOfResults

		    See GetFeedDictOfLSTM() method as well
		'''
		return [self._stateTensorOfLSTM_1]


	def GetFeedDictOfLSTM(self, BATCH_SIZE_, listOfPreviousStateValues_=None):
		'''
		      This function will return a dictionary that contained the PlaceHolder-Value map
		    of the LSTM states.
		      You can use this function as follows:
		    >> feed_dict = { netInput : batchOfImages }
		    >> feedDictOFLSTM = net.GetLSTM_Feed_Dict(BATCH_SIZE, listOfPreviousStateValues)
		    >> tupleOfOutputs = sess.run( [out] + net.GetListOfStatesTensorInLSTMs(),
						  feed_dict = feed_dict.update(feedDictOFLSTM) ) 
		    >> listOfOutputs = list(tupleOfOutputs)
		    >> output = listOfOutputs.pop(0)
		    >> listOfPreviousStateValues = listOfOutputs.pop(0)
		'''
		if listOfPreviousStateValues_ == None:
			'''
			    For the first time (or, the first of Unrolls), there's no previous state,
			    return zeros state.
			'''
			initialCellState = tuple( [np.zeros([BATCH_SIZE_, self._NUMBER_OF_NEURONS_IN_LSTM])] * 2 )
			initialCellState = tf.nn.rnn_cell.LSTMStateTuple(initialCellState[0], initialCellState[1])

			return {self._statePlaceHolderOfLSTM_1 : initialCellState }
		else:
			if len(listOfPreviousStateValues_) != 1:
				errorMessage = "len(listOfPreviousStateValues_) = " + str( len(listOfPreviousStateValues_) )
				errorMessage += "; However, the expected lenght is 1.\n"
				errorMessage += "\t Do you change the Network Structure, such as Add New LSTM?\n"
				errorMessage += "\t Or, do you add more tensor to session.run()?\n"

			return { self._statePlaceHolderOfLSTM_1 : listOfPreviousStateValues_[0] }


	def _buildOpticalFlowNet(self, inputTensor_):
		'''
		    The input shape = [b, u, g, w, h, c]
		    after Conv, shape = [b*u*g, w', h', c']
		    here, decouple the Group dimension, shape = [b*u, g * w' * h' * c']
		'''
		with tf.name_scope("OpticalFLow"):
			print("In OpticalFlow:")
			print("\t pool2.shape = ", inputTensor_.shape)  # shape = [b*u*g, 112, 112, 32]
			w, h, c = inputTensor_.shape[1:]  # 112, 112, 32
			out = tf.reshape( inputTensor_,
					  [self._batchSize * self._unrolledSize, dataSettings.GROUPED_SIZE,
					  w, h, c])  # [b*u, g, 112, 112, 32]
			out = tf.transpose(out, perm=[0, 2, 3, 4, 1])  # [b*u, 112, 112, 32, g]
			print("\t after transpose, out.shape = ", out.shape)
			out = tf.reshape( out, 
					  [self._batchSize * self._unrolledSize,
					   w, h, c * dataSettings.GROUPED_SIZE])
			print("\t before Conv2, out.shape = ", out.shape)  # shape = [b*u, 112, 112, 32*g]
			out = ConvLayer('Conv2', out, filterSize_=3, numberOfFilters_=64, stride_=1, padding_='SAME', isTrainable_=True)
			out, updateOp1 = BatchNormalization('BN2', out, isConvLayer_=True, isTraining_=self._isTraining,
								     currentStep_=self._trainingStep, isTrainable_=True)
			out = LeakyRELU('RELU2', out)
			out = MaxPoolLayer('Pool2', out, kernelSize_=2, stride_=2, padding_='SAME')

			out, updateOp2 = ResidualBlock('ResBlock', out, [128, 128, 64], isTraining_=self._isTraining,
							trainingStep_=self._trainingStep, activationType_="LEAKY_RELU", isTrainable_=True)

			out = MaxPoolLayer('Pool5', out, kernelSize_=2, stride_=2, padding_='SAME')

			print("\t after Pool5, out.shape = ", out.shape)  # shape = [b*u, 28, 28, 64]

			# Reshape to Concatenate with [b, u, 7, 7, 1024] tensor latter
			out = tf.reshape( out, [self._batchSize, self._unrolledSize, 7, 7, 1024] )

			self._dictOfInterestedActivations['OpticalFlow'] = out
			print("\t Fc final.shape = ", out.shape)
			updateOp = tf.group(updateOp1, updateOp2)
			return out, updateOp
