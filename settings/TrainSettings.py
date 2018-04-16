import tensorflow as tf
import settings.DataSettings as dataSettings

'''
      Following two variables control the shape of input
    data as the shape: [BATCH_SIZE*UNROLLED_SIZE, w, h, c].
    BATCH_SIZE: number of Videos in a batch.
    UNROLLED_SIZE: number of Frames in a Video.
      For the ConvNet part, the input will be the shape:
    [BATCH_SIZE*UNROLLED_SIZE, w, h, c].
      For the RNN part, the input will be the shape:
    [BATCH_SIZE, UNROLLED_SIZE, w, h, c] so that the
    tf.nn.rnn_cell.dynamic_rnn() can unroll the RNN.

    Note: Due to the fact that tf.TensorShape can't take
    PlaceHolder as input, currently this project does
    Not Support for dynamicly change the 'BATCH_SIZE' and
    the 'UNROLLED_SIZE'.
'''
BATCH_SIZE = 4
UNROLLED_SIZE = 40

PRETRAIN_MODEL_PATH_NAME = ""

'''
    If one want to finetune, insert the LastLayer to the following list.
    ex: NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT = ['Conv4', 'Conv5']
'''
NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT = []

MAX_TRAINING_EPOCH = 100

EPOCHS_TO_START_SAVE_MODEL = 20
PATH_TO_SAVE_MODEL = "temp/P1D19_1Fc_1LSTM-lr4"
MAX_TRAINING_SAVE_MODEL = MAX_TRAINING_EPOCH

def GetOptimizer(learningRate_):
	'''
	RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
	RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
	RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.
	return tf.train.RMSPropOptimizer(lr,
					 RMSPROP_DECAY,
					 momentum=RMSPROP_MOMENTUM,
					 epsilon=RMSPROP_EPSILON)
	'''
	return tf.train.AdamOptimizer(learning_rate=learningRate_)

'''
    Following list three different LearningRate decay methods:
	1. _stepLearningRate(),
	2. _exponentialDecayLearningRate()
	3. _polynomialDecayLearningRate()
'''
def _stepLearningRate(currentEpoch_):
	LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-4), (100, 1e-5) ]

	for eachPair in reversed(LIST_OF_EPOCH_LEARNING_RATE_PAIRS):
		if currentEpoch_ >= eachPair[0]:
			return eachPair[1]

	# If nothing matched, return the first pair.learningRate as default
	return LIST_OF_EPOCH_LEARNING_RATE_PAIRS[0][1] 


def _exponentialDecayLearningRate(currentStep_):
	'''
	    Exponential Decay:
		learningRate = INITIAL_LEARNING_RATE * DECAY_RATE ^ (currentStep_ / DECAY_STEP)
	'''
	INITIAL_LEARNING_RATE = 0.1
	DECAY_RATE = 0.16
	NUMBER_OF_EPOCHS_PER_DECAY = 30.0
	DECAY_STEP = int(dataSettings.NUMBER_OF_BATCHES_PER_EPOCH * NUMBER_OF_EPOCHS_PER_DECAY)
	learningRate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
						  global_step=currentStep_,
						  decay_step=DECAY_STEP,
						  decay_rate=DECAY_RATE,
						  staircase=False,
						  name='learningRate')
	return learningRate

def _polynomialDecayLearningRate(currentStep_):
	'''
	    Polynomial Decay:
		global_step = min(global_step, decay_steps)
		decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power)
					 + end_learning_rate
	'''
	learningRate = tf.train.polynomial_decay( learning_rate=0.1,
						  global_step=currentStep_,
						  decay_steps=MAX_STEPS,
						  end_learning_rate=1e-9,
						  power=4.0,
						  cycle=False,
						  name='learningRate'
						)
	return learningRate


def GetLearningRate(currentEpoch_=None, currentStep_=None):
	return _stepLearningRate(currentEpoch_)


