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
      The output of the total network will be the shape:
    [BATCH_SIZE, UNROLLED_SIZE, NUMBER_OF_CATEGORIES]
'''
BATCH_SIZE = 4
UNROLLED_SIZE = 40
#BATCH_SIZE = 40
#UNROLLED_SIZE = 1

PRETRAIN_MODEL_PATH_NAME = ""

'''
    If one want to finetune, insert the LastLayer to the following list.
    ex: NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT = ['Conv4', 'Conv5']
'''
NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT = []

MAX_TRAINING_EPOCH = 20

EPOCHS_TO_START_SAVE_MODEL = 4
PATH_TO_SAVE_MODEL = "temp/G2D19_Conv_1LSTM_flip_expLR"
MAX_TRAINING_SAVE_MODEL = MAX_TRAINING_EPOCH
PERFORM_DATA_AUGMENTATION = True

def GetOptimizer(learningRate_):
	return tf.train.AdamOptimizer(learning_rate=learningRate_)

'''
    Following list three different LearningRate decay methods:
	1. _stepLearningRate(),
	2. _exponentialDecayLearningRate()
'''
def _stepLearningRate(currentEpoch_, currentStep_):
	LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-4), (5, 1e-5) ]
	#LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-6), (20, 1e-7) ]

	for eachPair in reversed(LIST_OF_EPOCH_LEARNING_RATE_PAIRS):
		if currentEpoch_ >= eachPair[0]:
			return eachPair[1]

	# If nothing matched, return the first pair.learningRate as default
	return LIST_OF_EPOCH_LEARNING_RATE_PAIRS[0][1] 


def _exponentialDecayLearningRate(currentEpoch_, currentStep_):
	'''
	    Exponential Decay:
		learningRate = INITIAL_LEARNING_RATE * DECAY_RATE ^ (currentStep_ / DECAY_STEP)
	'''
	INITIAL_LEARNING_RATE = 1e-5
	#DECAY_RATE = 0.16
	DECAY_RATE = 0.9

	NUMBER_OF_BATCHES_PER_EPOCH = 125
	NUMBER_OF_EPOCHS_PER_DECAY = 1
	DECAY_STEP = int(NUMBER_OF_BATCHES_PER_EPOCH * NUMBER_OF_EPOCHS_PER_DECAY)

	learningRate = INITIAL_LEARNING_RATE * DECAY_RATE ** (currentStep_ / DECAY_STEP)

	return learningRate


def GetLearningRate(currentEpoch_=None, currentStep_=None):
#	return _stepLearningRate(currentEpoch_, currentStep_)
	return _exponentialDecayLearningRate(currentEpoch_, currentStep_=currentStep_)



#####################
# Advenced Settings #
#####################
WAITING_QUEUE_MAX_SIZE = 60
LOADED_QUEUE_MAX_SIZE = 30
NUMBER_OF_LOAD_DATA_THREADS=2
# WAITING_QUEUE_MAX_SIZE = 180
# LOADED_QUEUE_MAX_SIZE = 80
#NUMBER_OF_LOAD_DATA_THREADS=4
