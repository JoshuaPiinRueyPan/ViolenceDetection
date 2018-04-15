import tensorflow as tf
import numpy as np
import settings.LayerSettings as layerSettings
import settings.TrainSettings as trainSettings
import os

def L2_Regularizer(weightsTensor_):
	with tf.name_scope("L2_Regularizer"):
		if layerSettings.REGULARIZER_WEIGHTS_DECAY != None:
			weightDecay = tf.convert_to_tensor(layerSettings.REGULARIZER_WEIGHTS_DECAY,
							   dtype=weightsTensor_.dtype.base_dtype,
							   name='weightDecay')
			return tf.multiply(weightDecay, tf.nn.l2_loss(weightsTensor_), name='tf.multiply')

		else:
			return None

def Create_tfVariable(variableName_, initialValue_, isTrainable_, doesRegularize_=True):
	tf_variable = tf.Variable(initialValue_, dtype=layerSettings.FLOAT_TYPE, name=variableName_, trainable=isTrainable_)

	if (layerSettings.REGULARIZER_WEIGHTS_DECAY != None)and(doesRegularize_):
		regularizationLoss = L2_Regularizer(tf_variable)
		tf.losses.add_loss(regularizationLoss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

	return tf_variable


def CreateConvVariables(name_, filterSize_, inputChannels, numberOfFilters_, isTrainable_):
	with tf.name_scope(name_):
		weightsValue = tf.truncated_normal([filterSize_, filterSize_, inputChannels, numberOfFilters_],
						   mean=layerSettings.CONV_WEIGHTS_RNDOM_MEAN,
						   stddev=layerSettings.CONV_WEIGHTS_RNDOM_DEVIATION,
						   dtype=layerSettings.FLOAT_TYPE,
						   name="weightsValues")
		biasesValue = tf.truncated_normal([numberOfFilters_],
						  mean=layerSettings.CONV_BIASES_RNDOM_MEAN,
						  stddev=layerSettings.CONV_BIASES_RNDOM_DEVIATION,
						  dtype=layerSettings.FLOAT_TYPE,
						  name="biasesValues")
		weights = Create_tfVariable("weightsVariable", weightsValue, isTrainable_, doesRegularize_=True)
		biases = Create_tfVariable("biasesVariable", biasesValue, isTrainable_, doesRegularize_=False)
		return weights, biases


def CreateFcVariables(name_, numberOfInputs_, numberOfOutputs_, isTrainable_):
	with tf.name_scope(name_):
		weightsValue = tf.truncated_normal([numberOfInputs_, numberOfOutputs_],
						   mean=layerSettings.FC_WEIGHTS_RANDOM_MEAN,
						   stddev=layerSettings.FC_WEIGHTS_RANDOM_DEVIATION,
						   dtype=layerSettings.FLOAT_TYPE,
						   name="weightsValues")
		biasesValue = tf.truncated_normal([numberOfOutputs_],
						  mean=layerSettings.FC_BIASES_RANDOM_MEAN,
						  stddev=layerSettings.FC_BIASES_RANDOM_DEVIATION,
						  dtype=layerSettings.FLOAT_TYPE,
						  name="biasesValues")
		weights = Create_tfVariable("weightsVariable", weightsValue, isTrainable_, doesRegularize_=True)
		biases = Create_tfVariable("biasesVariable", biasesValue, isTrainable_, doesRegularize_=False)
		return weights, biases


def CountElementsInOneFeatureMap(inputTensor_):
	'''
	   This function calculate number of elements in an image.
	   For example, if you have a feature map with (b, w, h, c)
	   this function will return w*h*c.  i.e. without consider
	   the batch dimension.
	'''
	featureMapShape = inputTensor_.shape[1:]
	return int( np.prod(featureMapShape) )


