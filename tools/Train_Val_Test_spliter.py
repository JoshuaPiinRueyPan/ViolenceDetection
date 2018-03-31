'''
    This script will save 3 files: train.txt, val.txt, test.txt.
    each file has the following format:
	--------------------------------------------------------------
	PathFileNameOfFile <Tab> FightStartFrame <Tab> FightEndFrame
	ex:
		data/video/video_1	0.0	INF
		data/video/video_2	INF	INF
	--------------------------------------------------------------
'''
#=====================================================================================
#    User Settings (The following variables should be adjust to your need.)
#=====================================================================================
LIST_OF_NO_FIGHT_DIR = ["data/Bermejo/hockey/noFights/"]
LIST_OF_FIGHT_DIR = ["data/Bermejo/hockey/fights/"]

VAL_SET_RATIO = 0.1
TEST_SET_RATIO = 0.4   # NUMBER_OF_TEST_SET = NUMBER_OF_TOTAL_DATA * TEST_SET_RATIO

PATH_TO_SAVE_SPLITED_DATASET = "."

#=====================================================================================
#    End of User Settings
#=====================================================================================

def PrintHelp():
	print("Usage: $(ThisScript)")
	print("\t Note: You should specified some variables in the top of this script")

import os

def AccumulateAllVideoFromDifferentDir(LIST_OF_DIR_THAT_CONTAINS_VIDEOS_):
	listOfVideos = []
	for eachDir in LIST_OF_DIR_THAT_CONTAINS_VIDEOS_:
		listOfVideos += os.listdir(eachDir)

	return listOfVideos

def AppendLabelToEachData(LIST_OF_DATA_, isFighting_):
	listOfDataWithLabel = []
	for eachData in LIST_OF_DATA_:
		if isFighting_:
			eachData += "\t0.0\tINF"

		else:
			eachData += "\tINF\tINF"

		listOfDataWithLabel.append(eachData)

	return listOfDataWithLabel

import random
def Split_Train_Val_Test_Data(LIST_OF_VIDEOS_):
	random.shuffle(LIST_OF_VIDEOS_)
	NUMBER_OF_TOTAL_DATA = len(LIST_OF_VIDEOS_)
	NUMBER_OF_TEST_VIDEOS = int(NUMBER_OF_TOTAL_DATA * TEST_SET_RATIO)
	NUMBER_OF_VAL_VIDEOS = int(NUMBER_OF_TOTAL_DATA * VAL_SET_RATIO)

	listOfTestVideos = LIST_OF_VIDEOS_[ : NUMBER_OF_TEST_VIDEOS]
	listOfValVideos = LIST_OF_VIDEOS_[NUMBER_OF_TEST_VIDEOS : (NUMBER_OF_TEST_VIDEOS+NUMBER_OF_VAL_VIDEOS)]
	listOfTrainVideos = LIST_OF_VIDEOS_[(NUMBER_OF_TEST_VIDEOS+NUMBER_OF_VAL_VIDEOS) : ]

	return listOfTrainVideos, listOfValVideos, listOfTestVideos

def WriteDataSetToFile(LIST_OF_DATA_, targetFileName_):
	with open(targetFileName_, 'w') as fileWriter:
		for eachData in LIST_OF_DATA_:
			fileWriter.write(eachData + "\n")
		

if __name__ == "__main__":
	listOfNoFightVideos = AccumulateAllVideoFromDifferentDir(LIST_OF_NO_FIGHT_DIR)
	listOfNoFightVideos = AppendLabelToEachData(listOfNoFightVideos, isFighting_=False)
	trainNoFightVideos, valNoFightVideos, testNoFightVideos = Split_Train_Val_Test_Data(listOfNoFightVideos)

	listOfFightVideos = AccumulateAllVideoFromDifferentDir(LIST_OF_FIGHT_DIR)
	listOfFightVideos = AppendLabelToEachData(listOfFightVideos, isFighting_=True)
	trainFightVideos, valFightVideos, testFightVideos = Split_Train_Val_Test_Data(listOfFightVideos)

	listOfTrainData = trainNoFightVideos + trainFightVideos
	random.shuffle(listOfTrainData)
	WriteDataSetToFile(listOfTrainData, os.path.join(PATH_TO_SAVE_SPLITED_DATASET, 'train.txt') )

	listOfValData = valNoFightVideos + valFightVideos
	random.shuffle(listOfValData)
	WriteDataSetToFile(listOfValData, os.path.join(PATH_TO_SAVE_SPLITED_DATASET, 'val.txt') )

	listOfTestData = testNoFightVideos + testFightVideos
	random.shuffle(listOfTestData)
	WriteDataSetToFile(listOfTestData, os.path.join(PATH_TO_SAVE_SPLITED_DATASET, 'test.txt') )

