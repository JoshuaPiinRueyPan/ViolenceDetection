from abc import ABCMeta, abstractmethod

class NetworkBase:
	__metaclass__ = ABCMeta
	@abstractmethod
	def Build(self):
		pass

	@property
	@abstractmethod
	def logitsOp(self):
		pass

	@property
	@abstractmethod
	def updateOp(self):
		pass

	@property
	def dictionaryOfInterestedActivations(self):
		'''
		    The implementations must define 'self._dictOfInterestedActivations'.
		    If there's no interested activations, simply:
			self._dictOfInterestedActivations = {}
		'''
		return self._dictOfInterestedActivations


	'''
	    The following two functions: 'GetListOfStatesTensorInLSTMs()' & 'GetLSTM_Feed_Dict()'
	    are used to Decouple the implementation of Net and the Evaluator.
	'''

	@abstractmethod
	def GetListOfStatesTensorInLSTMs(self):
		'''
		      This function should return the LSTM States' Tensor.  And the user will sess.run()
		    these tensor and get its values (referered as 'listOfPreviousStateValues').
		      For the next batch, the 'listOfPreviousStateValues' will be inserted to the
		    PlaceHolder of LSTM Previous States.  So that the LSTM can have the previous memory.
		'''
		pass

	@abstractmethod
	def GetFeedDictOfLSTM(self, BATCH_SIZE_, listOfPreviousStateValues_=None):
		'''
		    This function should return a feed_dict that contained the Map between
		    (PlaceHolders of LSTM Previous States) and ('listOfPreviousStateValues').
		'''
		pass
