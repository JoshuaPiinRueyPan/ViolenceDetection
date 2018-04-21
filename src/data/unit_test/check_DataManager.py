#!/usr/bin/python3

from src.data.DataManager import *
import settings.DataSettings as dataSettings
import cv2
import numpy as np
import time
import src.data.ImageUtils as ImageUtils

#PATH_TO_DATA = 'src/data/unit_test/videos.txt'
#PATH_TO_DATA = 'data/val.txt'
PATH_TO_DATA = 'data/train.txt'

def DrawInfo(targetImage_, listOfInfoToDisplay):
	xLeft = 0
	yTop = 0

	xRight = 50

	textSpace = 20
	currentTextPos = yTop + textSpace

	for i, eachInfo in enumerate(listOfInfoToDisplay):
		if i != 0:
			currentTextPos += textSpace
		cv2.putText(	targetImage_,
				eachInfo,
				(xLeft + 5, currentTextPos),
				cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=0.5,
				color=(255, 255, 255),
				thickness=1,
				lineType=cv2.LINE_AA)
	return targetImage_

def Check_TrainDataManager():
	pauseLoadData = False

	print("Start reading videos...")
	dataManager = TrainDataManager(PATH_TO_DATA)
	print("Read videos finished.")

	while True:
		'''
		    The following Info should be extracted before calling 'AssignBatchData()'
		'''
		listOfBatchInfo = []
		listOfBatchInfo.append('dataManager.epoch='+str(dataManager.epoch))
		listOfBatchInfo.append('dataManager.step='+str(dataManager.step))

		batchData = BatchData()
		startGetBatchTime = time.time()
		dataManager.AssignBatchData(batchData)
		finishGetBatchTime = time.time()
		print("GetBatchTime = ", finishGetBatchTime - startGetBatchTime)

		info = dataManager.GetQueueInfo()
		print("\t" + info + "\n")

		batchData.batchOfImages = batchData.batchOfImages.reshape([batchData.batchSize * batchData.unrolledSize
											       * batchData.groupedSize,
									   dataSettings.IMAGE_SIZE,
									   dataSettings.IMAGE_SIZE,
									   dataSettings.IMAGE_CHANNELS])
		batchData.batchOfLabels = batchData.batchOfLabels.reshape([batchData.batchSize * batchData.unrolledSize,
									   dataSettings.NUMBER_OF_CATEGORIES])

		i = 0
		while i < batchData.batchOfImages.shape[0]:
			currentImage = batchData.batchOfImages[i]
			currentImage = ImageUtils.ConvertImageFrom_NetInput_to_CV(currentImage)
			currentImage = cv2.resize(currentImage, (500, 500))
			currentLabel = batchData.batchOfLabels[ int(i/dataSettings.GROUPED_SIZE) ]

			listOfInfoToDisplay = []
			listOfInfoToDisplay += listOfBatchInfo
			listOfInfoToDisplay.append('i = '+str(i))
			listOfInfoToDisplay.append('batchImages.shape = ' + str(batchData.batchOfImages.shape))
			listOfInfoToDisplay.append('label = '+str(currentLabel))
			resultImage = DrawInfo(currentImage, listOfInfoToDisplay)

			cv2.imshow("Result", resultImage)

			userResponse = cv2.waitKey(0)
			if userResponse == ord('n'):
				i += 1

			elif userResponse == ord('l'):
				i = batchData.batchOfImages.shape[0] - 1

			elif userResponse == ord('p'):
				pauseLoadData = not pauseLoadData
				if pauseLoadData:
					dataManager.Pause()
				else:
					dataManager.Continue()

			elif userResponse == ord('q'):
				dataManager.Stop()
				raise StopIteration()



def Check_EvalDataManager():
	pauseLoadData = False

	print("Start reading videos...")
	dataManager = EvaluationDataManager(PATH_TO_DATA)
	print("Read videos finished.")

	while True:
		'''
		    The following Info should be extracted before calling 'AssignBatchData()'
		'''
		listOfBatchInfo = []
		listOfBatchInfo.append('dataManager.isAllDataTraversed='+str(dataManager.isAllDataTraversed))
		listOfBatchInfo.append('dataManager.isNewVideo='+str(dataManager.isNewVideo))

		startCreateTime = time.time()
		batchData = BatchData()
		endCreateTime = time.time()
		print("Create time = ", endCreateTime - startCreateTime)

		startGetBatchTime = time.time()
		dataManager.AssignBatchData(batchData)
		finishGetBatchTime = time.time()
		print("GetBatchTime = ", finishGetBatchTime - startGetBatchTime, "\n")

		batchData.batchOfImages = batchData.batchOfImages.reshape([batchData.batchSize * batchData.unrolledSize
											       * batchData.groupedSize,
									   dataSettings.IMAGE_SIZE,
									   dataSettings.IMAGE_SIZE,
									   dataSettings.IMAGE_CHANNELS])

		batchData.batchOfLabels = batchData.batchOfLabels.reshape([batchData.batchSize * batchData.unrolledSize,
									   dataSettings.NUMBER_OF_CATEGORIES])



		info = dataManager.GetQueueInfo()
		print("\t" + info + "\n")
		i = 0
		while i < batchData.batchOfImages.shape[0]:
			currentImage = batchData.batchOfImages[i]
			currentImage = ImageUtils.ConvertImageFrom_NetInput_to_CV(currentImage)
			currentImage = cv2.resize(currentImage, (500, 500))
			
			currentLabel = batchData.batchOfLabels[ int(i/dataSettings.GROUPED_SIZE) ]

			listOfInfoToDisplay = []
			listOfInfoToDisplay += listOfBatchInfo
			listOfInfoToDisplay.append('i = '+str(i))
			listOfInfoToDisplay.append('batchImages.shape = ' + str(batchData.batchOfImages.shape))
			listOfInfoToDisplay.append('label = '+str(currentLabel))
			resultImage = DrawInfo(currentImage, listOfInfoToDisplay)

			cv2.imshow("Result", resultImage)

			userResponse = cv2.waitKey(0)
			if userResponse == ord('n'):
				i += 1

			elif userResponse == ord('l'):
				i = batchData.batchOfImages.shape[0] - 1

			elif userResponse == ord('p'):
				pauseLoadData = not pauseLoadData
				if pauseLoadData:
					dataManager.Pause()
				else:
					dataManager.Continue()

			elif userResponse == ord('q'):
				dataManager.Stop()
				raise StopIteration()



if __name__ == '__main__':
	userSelectedMode = int( input("Which DataManager do you want to test? (0:train / 1:eval)  ") )
	if userSelectedMode == 0:
		Check_TrainDataManager()

	elif userSelectedMode == 1:
		Check_EvalDataManager()
	else:
		raise ValueError("You should specified which kind of DataManager you want to test:"
				 +" 'train' or 'eval'")
