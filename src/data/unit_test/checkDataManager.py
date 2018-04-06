from src.data.DataManager import *
import settings.DataSettings as dataSettings
import cv2
import numpy as np
import time

#PATH_TO_DATA = 'src/data/unit_test/videos.txt'
PATH_TO_DATA = 'data/train.txt'
#targetDataManager = 'train'
targetDataManager = 'eval'

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

def convertNetInputToCV_Format(netInputImage_):
	cvImage = (netInputImage_ + 1.0) * (255.0/2.)
	cvImage = cv2.cvtColor(cvImage.astype(np.uint8), cv2.COLOR_RGB2BGR)
	return cvImage

def Check_TrainDataManager():
	pauseLoadData = False

	print("Start reading videos...")
	dataSettings.PATH_TO_TRAIN_SET_LIST = PATH_TO_DATA
	dataManager = TrainDataManager()
	print("Read videos finished.")

	while True:
		'''
		    The following Info should be extracted before calling 'GetBatchOfData()'
		'''
		listOfBatchInfo = []
		listOfBatchInfo.append('dataManager.epoch='+str(dataManager.epoch))
		listOfBatchInfo.append('dataManager.step='+str(dataManager.step))

		batchData = BatchData()
		startGetBatchTime = time.time()
		dataManager.GetBatchOfData(batchData)
		#batchData = dataManager.GetBatchOfData()
		finishGetBatchTime = time.time()
		print("GetBatchTime = ", finishGetBatchTime - startGetBatchTime)

		info = dataManager.GetQueueInfo()
		print("\t" + info + "\n")
		i = 0
		while i < batchData.batchOfImages.shape[0]:
			currentImage = batchData.batchOfImages[i]
			currentImage = convertNetInputToCV_Format(currentImage)
			currentLabel = batchData.batchOfLabels[i]

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
		    The following Info should be extracted before calling 'GetBatchOfData()'
		'''
		listOfBatchInfo = []
		listOfBatchInfo.append('dataManager.isAllDataTraversed='+str(dataManager.isAllDataTraversed))
		listOfBatchInfo.append('dataManager.isNewVideo='+str(dataManager.isNewVideo))

		startCreateTime = time.time()
		batchData = BatchData()
		endCreateTime = time.time()
		print("Create time = ", endCreateTime - startCreateTime)

		startGetBatchTime = time.time()
		dataManager.GetBatchOfData(batchData)
		finishGetBatchTime = time.time()
		print("GetBatchTime = ", finishGetBatchTime - startGetBatchTime, "\n")

		info = dataManager.GetQueueInfo()
		print("\t" + info + "\n")
		i = 0
		while i < batchData.batchOfImages.shape[0]:
			currentImage = batchData.batchOfImages[i]
			currentImage = convertNetInputToCV_Format(currentImage)
			currentLabel = batchData.batchOfLabels[i]

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
	if targetDataManager == 'train':
		Check_TrainDataManager()

	elif targetDataManager == 'eval':
		Check_EvalDataManager()
	else:
		raise ValueError("You should specified which kind of DataManager you want to test:"
				 +" 'train' or 'eval'")
