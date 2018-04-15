import tensorflow as tf
import src.layers.LayerHelper as LayerHelper
import settings.LayerSettings as layerSettings

def ConvLayer(name_, inputTensor_, filterSize_, numberOfFilters_, stride_=1, padding_='SAME', isTrainable_=True):
	with tf.name_scope(name_):
		inputChannels = int(inputTensor_.shape[3])
		weights, biases = LayerHelper.CreateConvVariables("CreateConvVariables", filterSize_,
								  inputChannels, numberOfFilters_, isTrainable_)
		convTensor = tf.nn.conv2d(inputTensor_, weights, strides=[1, stride_, stride_, 1],
					  padding=padding_, name="tf.nn.conv2d")
		outputTensor = tf.nn.bias_add(convTensor, biases, name="tf.nn.bias_add")

		if layerSettings.DOES_SHOW_CONV_SUMMARY:
			tf.summary.histogram('weightsSummary', weights)
			tf.summary.histogram('biasesSummary', biases)

		return outputTensor

def FullyConnectedLayer(name_, inputTensor_, numberOfOutputs_, isTrainable_=True):
	with tf.name_scope(name_):
		numberOfInputs = LayerHelper.CountElementsInOneFeatureMap(inputTensor_)
		inputTensor_ = tf.reshape(inputTensor_, [-1, numberOfInputs], name="flatten")
		weights, biases = LayerHelper.CreateFcVariables("CreateFcVariables", numberOfInputs, numberOfOutputs_, isTrainable_)
		fc = tf.matmul(inputTensor_, weights, name="tf.matmul")
		output = tf.nn.bias_add(fc, biases, name="tf.nn.bias_add")

		if layerSettings.DOES_SHOW_FC_SUMMARY:
			tf.summary.histogram('weightsSummary', weights)
			tf.summary.histogram('biasesSummary', biases)

		return output

def LeakyRELU(name_, inputTensor_):
	return tf.maximum(layerSettings.LEAKY_RELU_FACTOR*inputTensor_, inputTensor_, name=name_)

def SetActivation(name_, inputTensor_, activationType_):
	if activationType_.upper() == "RELU":
		return tf.nn.relu(inputTensor_, name_)

	elif activationType_.upper() == "LEAKY_RELU":
		return LeakyRELU(name_, inputTensor_)

	else:
		errorMessage = "You may add a new Activation type: '" + activationType_.upper() + "'\n"
		errorMessage += "but not implement in SetActivation()"
		raise ValueError(errorMessage)

def AlexNorm(name_, inputTensor, lsize=4):
	return tf.nn.lrn(inputTensor, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name_)

def MaxPoolLayer(name_, inputTensor_, kernelSize_=2, stride_=2, padding_='SAME'):
	return tf.nn.max_pool(inputTensor_, ksize=[1, kernelSize_, kernelSize_, 1],
				 strides=[1, stride_, stride_, 1], padding=padding_, name=name_)


def AvgPoolLayer(name_, inputTensor_, kernelSize_=2, stride_=2, padding_='SAME'):
	return tf.nn.avg_pool(inputTensor_, ksize=[1, kernelSize_, kernelSize_, 1],
				 strides=[1, stride_, stride_, 1],
				 padding=padding_,
				 name=name_)

def BatchNormalization(name_, inputTensor_, isConvLayer_, isTraining_, currentStep_, isTrainable_=True):
	with tf.name_scope(name_):
		currentBatchMean = None
		currentBatchVariance = None
		outputChannels = None
		if isConvLayer_:
			currentBatchMean, currentBatchVariance = tf.nn.moments(inputTensor_, [0, 1, 2], name="tf.nn.moments")
		else:
			currentBatchMean, currentBatchVariance = tf.nn.moments(inputTensor_, [0], name="tf.nn.moments")

		'''
		    The mean & variance of variables in Testing will be calculated by following methods:
			decay = min(decay, (1 + step)/(10+step) )
			totalMean = (1 - decay)*currentBatchMean + decay*totalMean
		    in which totalMean also called 'shadow variable' of currentBatchMean.

		    Note: It's meaningless to take the first step (which has been Random Initialized)
		    for average.  Therefore, one should add the training step to scale the previous
		    steps down.
		'''
		averageCalculator = tf.train.ExponentialMovingAverage(decay=layerSettings.BATCH_NORMALIZATION_MOVING_AVERAGE_DECAY_RATIO,
								      num_updates=currentStep_, name="EMA")
		updateVariablesOperation = averageCalculator.apply( [currentBatchMean, currentBatchVariance] )

		totalMean = tf.cond(isTraining_,
				    lambda: currentBatchMean, lambda: averageCalculator.average(currentBatchMean),
				    name="averagedMean")

		totalVariance = tf.cond(isTraining_,
					lambda: currentBatchVariance, lambda: averageCalculator.average(currentBatchVariance),
					name="averagedVariance")
		'''
		    Following will calculate BatchNormalization by:
			X' = gamma * (X - mean)/sqrt(variance**2 + epsillon) + betta
		'''
		outputChannels = int(inputTensor_.shape[-1])
		gamma = LayerHelper.Create_tfVariable("Gamma", tf.ones([outputChannels]), isTrainable_, doesRegularize_=False)
		betta = LayerHelper.Create_tfVariable("Betta", tf.zeros([outputChannels]), isTrainable_, doesRegularize_=False)
		epsilon = 1e-5
		outputTensor = tf.nn.batch_normalization(inputTensor_, mean=totalMean, variance=totalVariance, offset=betta,
							 scale=gamma, variance_epsilon=epsilon,
							 name="tf.nn.batch_norm")
		return outputTensor, updateVariablesOperation
