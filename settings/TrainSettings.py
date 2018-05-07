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
#UNROLLED_SIZE = 2

PRETRAIN_MODEL_PATH_NAME = ""
#PRETRAIN_MODEL_PATH_NAME = "temp/G2D19_P2OF_ResHB_1LSTM_dataAug_expLR/save_epoch_14/ViolenceNet.ckpt"

'''
    If one want to finetune, insert the LastLayer to the following list.
    ex: NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT = ['Conv4', 'Conv5']
'''
NAME_SCOPES_NOT_TO_RECOVER_FROM_CHECKPOINT = []

MAX_TRAINING_EPOCH = 30

EPOCHS_TO_START_SAVE_MODEL = 1
PATH_TO_SAVE_MODEL = "temp/G2D19_P2OF_ResHB_1LSTM_dataAug_expLR"
MAX_TRAINING_SAVE_MODEL = MAX_TRAINING_EPOCH
PERFORM_DATA_AUGMENTATION = True

def GetOptimizer(learningRate_):
	return tf.train.AdamOptimizer(learning_rate=learningRate_)

'''
    Following list three different LearningRate decay methods:
	1. _stairLearningRate(),
	2. _exponentialDecayLearningRate()
	3. _polynomialDecayLearningRate()
'''
def _stairLearningRate(currentEpoch_, currentStep_):
	#LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-4), (5, 1e-5) ]
	LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 1e-6), (15, 5e-7), (25, 1e-7) ]
	#LIST_OF_EPOCH_LEARNING_RATE_PAIRS = [ (0, 3e-6), (5, 2.5e-6), (10, 2e-6), (15, 1.5e-6), (20, 1e-6) ]

	for eachPair in reversed(LIST_OF_EPOCH_LEARNING_RATE_PAIRS):
		if currentEpoch_ >= eachPair[0]:
			return eachPair[1]

	# If nothing matched, return the first pair.learningRate as default
	return LIST_OF_EPOCH_LEARNING_RATE_PAIRS[0][1] 


def _exponentialDecayLearningRate(currentEpoch_, currentStep_):
	'''
	    Exponential Decay:
		learningRate = INITIAL_LEARNING_RATE * DECAY_RATE ^ (currentStep_ / DECAY_STEP) + END_LEARNING_RATE
	'''
	INITIAL_LEARNING_RATE = 1e-5
	DECAY_RATE = 0.9
	NUMBER_OF_BATCHES_PER_EPOCH = 250
	NUMBER_OF_EPOCHS_PER_DECAY = 1
	DECAY_STEP = int(NUMBER_OF_BATCHES_PER_EPOCH * NUMBER_OF_EPOCHS_PER_DECAY)
	END_LEARNING_RATE = 0.0

	learningRate = INITIAL_LEARNING_RATE * DECAY_RATE ** (currentStep_ / DECAY_STEP) + END_LEARNING_RATE

	return learningRate

def _polynomialDecayLearningRate(currentEpoch_, currentStep_):
	'''
	    Polynomial Decay:
		step = min(currentStep_, MAX_STEPS)
		learningRate = (START_LEARNING_RATE - END_LEARNING_RATE) * (1 - step/MAX_STEPS)^(POWER) + END_LEARNING_RATE
	'''
	START_LEARNING_RATE = 2e-6
	END_LEARNING_RATE = 1e-7
	MAX_STEPS = MAX_TRAINING_EPOCH * 125
	POWER = 4



def GetLearningRate(currentEpoch_, currentStep_):
#	return _stairLearningRate(currentEpoch_, currentStep_)
	return _exponentialDecayLearningRate(currentEpoch_, currentStep_=currentStep_)



#####################
# Advenced Settings #
#####################
'''
    Following settings depend on (BATCH_SIZE, UNROLLED_SIZE, PERFORM_DATA_AUGMENTATION):
    if (4, 40, False), Recommend values:
	WAITING_QUEUE_MAX_SIZE = 60
	LOADED_QUEUE_MAX_SIZE = 30
	NUMBER_OF_LOAD_DATA_THREADS=2

    if (4, 40, True), Recommend values:
	WAITING_QUEUE_MAX_SIZE = 180
	LOADED_QUEUE_MAX_SIZE = 80
	NUMBER_OF_LOAD_DATA_THREADS=2

    if (40, 1, False), Recommend values:
	WAITING_QUEUE_MAX_SIZE = 180
	LOADED_QUEUE_MAX_SIZE = 80
	NUMBER_OF_LOAD_DATA_THREADS=4

    if (40, 1, True), Recommend values:
	WAITING_QUEUE_MAX_SIZE = 180
	LOADED_QUEUE_MAX_SIZE = 80
	NUMBER_OF_LOAD_DATA_THREADS=4

     Note: The "Averaged GetBatch Time" that printed while you train an epoch, should be
	   smaller than 0.001(s). Otherwise, increase NUMBER_OF_LOAD_DATA_THREADS.
'''
WAITING_QUEUE_MAX_SIZE = 180
LOADED_QUEUE_MAX_SIZE = 80
NUMBER_OF_LOAD_DATA_THREADS=4

MAX_GRADIENT_VALUE = 5.0
MIN_GRADIENT_VALUE = -5.0
