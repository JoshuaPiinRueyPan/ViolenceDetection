from src.layers.LayerHelper import *
import settings.LayerSettings as layerSettings
import tensorflow as tf


def LSTM(name_, inputTensor_, numberOfOutputs_, isTraining_, dropoutProb_=None):
	with tf.name_scope(name_):
		cell = tf.nn.rnn_cell.LSTMCell(num_units=numberOfOutputs_,
						 use_peepholes=True,
						 initializer=layerSettings.LSTM_INITIALIZER,
						 forget_bias=1.0,
						 state_is_tuple=True,
						 activation=tf.nn.tanh,
						 name=name_+"_cell")

		if dropoutProb_ != None:
			dropoutProbTensor = tf.cond(isTraining_, lambda: 0.5, lambda: 1.0)
			cell = tf.nn.rnn_cell.DropoutWrapper(cell,
							     input_keep_prob=dropoutProbTensor,
							     output_keep_prob=dropoutProbTensor)

		statePlaceHolder = tf.nn.rnn_cell.LSTMStateTuple( tf.placeholder(layerSettings.FLOAT_TYPE, [None, numberOfOutputs_]),
								  tf.placeholder(layerSettings.FLOAT_TYPE, [None, numberOfOutputs_]) )

		outputTensor, stateTensor = tf.nn.dynamic_rnn(	cell=cell,
								initial_state=statePlaceHolder,
								inputs=inputTensor_)

		# Add Regularization Loss
		for eachVariable in tf.trainable_variables():
			if name_ in eachVariable.name:
				if ('bias' not in eachVariable.name)and(layerSettings.REGULARIZER_WEIGHTS_DECAY != None):
					regularizationLoss = L2_Regularizer(eachVariable)
					tf.losses.add_loss(regularizationLoss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
					

	return outputTensor, stateTensor, statePlaceHolder

