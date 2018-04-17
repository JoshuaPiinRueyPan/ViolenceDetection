#!/usr/bin/python3

from src.ViolenceDetector import *

def Check_OutputSmootherByBinaryArray(array_, answer_):
	outputSmoother = OutputSmoother()
	for i in range(len(array_)):
		currentResult = outputSmoother.Smooth(array_[i])
		if currentResult != answer_[i]:
			errorMessage = "array_ = " + str(array_) + "\n"
			errorMessage = "answer_ = " + str(answer_) + "\n"
			errorMessage = "while in index = " + str(i) + ":\n"
			errorMessage = "\t smooth(array_) = " + str(currentResult) + "; "
			errorMessage = " answer_[i] = " + str(answer_[i])
			
			raise ValueError(errorMessage)
	print("-----------------------------------------------------------------------------------------")
	print("array = \t" + str(array_) )
	print("answer = \t" + str(answer_) )
	print("\t check passed.\n")

if __name__ == '__main__':
	arrayA = [	False,	True,	True,	True,	False,	True,	False,	False,	False ]
	answerA = [	False,	False,	False,	True,	True,	True,	True,	True,	False ]
	Check_OutputSmootherByBinaryArray(arrayA, answerA)

	arrayB = [	True,	True,	False,	True,	True,	False,	True,	False,	True ]
	answerB = [	False,	False,	False,	False,	False,	False,	False,	False,	False ]
	Check_OutputSmootherByBinaryArray(arrayB, answerB)

	arrayC = [	True,	False,	True,	True,	True,	True,	True,	False,	False,	False ]
	answerC = [	False,	False,	False,	False,	True,	True,	True,	True,	True,	False ]
	Check_OutputSmootherByBinaryArray(arrayC, answerC)

	arrayD = [	True,	True,	True,	True,	False,	True,	False,	False,	True ]
	answerD = [	False,	False,	True,	True,	True,	True,	True,	True,	True ]
	Check_OutputSmootherByBinaryArray(arrayD, answerD)
