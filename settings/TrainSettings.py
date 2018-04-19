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

EPOCHS_TO_START_SAVE_MODEL = 1
PATH_TO_SAVE_MODEL = "temp/G2D19_1Fc_1LSTM_noDataAug_expLR"
MAX_TRAINING_SAVE_MODEL = MAX_TRAINING_EPOCH
PERFORM_DATA_AUGMENTATION = False

def GetOptimizer(learningRate_):
	return tf.train.AdamOptimizer(learning_rate=learningRate_)

'''
    Following list three different LearningRate decay methods:
	1. _stepLearningRate(),
	2. _exponentialDecayLearningRate()
'''
def _stepLearningRate(currentEpoch_):
	#LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-4), (5, 1e-5) ]
	LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-6), (15, 1e-7) ]

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
	INITIAL_LEARNING_RATE = 1e-5
	DECAY_RATE = 0.16

	NUMBER_OF_BATCHES_PER_EPOCH = 125
	NUMBER_OF_EPOCHS_PER_DECAY = 1
	DECAY_STEP = int(NUMBER_OF_BATCHES_PER_EPOCH * NUMBER_OF_EPOCHS_PER_DECAY)

	learningRate = INITIAL_LEARNING_RATE * DECAY_RATE ** (currentStep_ / DECAY_STEP)

	return learningRate


def GetLearningRate(currentEpoch_=None, currentStep_=None):
#	return _stepLearningRate(currentEpoch_)
	return _exponentialDecayLearningRate(currentStep_=currentStep_)



#####################
# Advenced Settings #
#####################
WAITING_QUEUE_MAX_SIZE = 120
LOADED_QUEUE_MAX_SIZE = 30
NUMBER_OF_LOAD_DATA_THREADS=1
# WAITING_QUEUE_MAX_SIZE = 180
# LOADED_QUEUE_MAX_SIZE = 80
#NUMBER_OF_LOAD_DATA_THREADS=4

'''
    If TrainLoss > LOSS_THRESHOLD_TO_SAVE_DEBUG_IMAGE,
    and epoch > EPOCH_TO_START_SAVEING_DEBUG_IMAGE,
    the program will automatically save that batch
    images.  This offenly used to make sure if the
    Augmented Data been driven too far.
'''
LOSS_THRESHOLD_TO_SAVE_DEBUG_IMAGE = 0.5
EPOCH_TO_START_SAVEING_DEBUG_IMAGE = 3
