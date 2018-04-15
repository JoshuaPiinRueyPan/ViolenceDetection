import tensorflow as tf
import settings.DataSettings as dataSettings

BATCH_SIZE = 3
UNROLLED_SIZE = 40

PRETRAIN_MODEL_PATH_NAME = ""

'''
    If one want to finetune, insert the LastLayer to the following list.
    ex: NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT = ['Conv4', 'Conv5']
'''
NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT = []

MAX_TRAINING_EPOCH = 50

EPOCHS_TO_START_SAVE_MODEL = 20
PATH_TO_SAVE_MODEL = "temp/models/DarkNet19/dataAug/32_leaky_train-2"
#PATH_TO_SAVE_MODEL = "temp/models/DarkNet19/dataAug/biasAdd_13"
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
	LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-4), (15, 1e-5), (40, 1e-6) ]
	#LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-3), (20, 1e-4) ]
	#LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-3), (30, 1e-4), (80, 1e-5) ]
	#LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-4), (30, 1e-5), (50, 1e-6) ]

	for eachPair in reversed(LIST_OF_EPOCH_LEARNING_RATE_PAIRS):
		if currentEpoch_ >= eachPair[0]:
			return eachPair[1]

	# If nothing matched, return the first pair.learningRate as default
	return trainSettings.LIST_OF_EPOCH_LEARNING_RATE_PAIRS[0][1] 


def _exponentialDecayLearningRate(currentEpoch_):
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

def _polynomialDecayLearningRate(currentEpoch_):
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


def GetLearningRate(currentStep_):
	return _stepLearningRate(currentStep_)


