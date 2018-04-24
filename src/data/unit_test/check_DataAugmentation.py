#!/usr/bin/python3

import os
import sys
import cv2
import skvideo.io
import numpy as np
import src.data.DataAugmenter as DataAugmenter
import src.data.ImageUtils as ImageUtils

'''
    Change the following variable if you have different data arrangement
'''
LIST_OF_VIDEOS = [
	"data/Bermejo/hockey/noFights/no301_xvid.avi",
	"data/Bermejo/hockey/fights/fi114_xvid.avi",
	"data/Bermejo/hockey/fights/fi402_xvid.avi",
	"data/Bermejo/hockey/noFights/no353_xvid.avi",
	"data/Bermejo/hockey/fights/fi261_xvid.avi",
	"data/Bermejo/hockey/noFights/no317_xvid.avi",
	"data/Bermejo/hockey/noFights/no361_xvid.avi",
	"data/Bermejo/hockey/fights/fi203_xvid.avi",
	"data/Bermejo/hockey/noFights/no119_xvid.avi",
	"data/Bermejo/hockey/noFights/no411_xvid.avi",
	"data/Bermejo/hockey/noFights/no388_xvid.avi",
	"data/Bermejo/hockey/fights/fi71_xvid.avi",
	"data/Bermejo/hockey/noFights/no434_xvid.avi",
	"data/Bermejo/hockey/fights/fi185_xvid.avi",
	"data/Bermejo/hockey/fights/fi349_xvid.avi",
	"data/Bermejo/hockey/noFights/no207_xvid.avi"

]


def PrintHelp():
	print("Usage:")
	print("\t $(ThisScript)")
	print("\t\t Note: If you're not put your data in 'data/Bermejo/hockey', you should edit this script")
	print("\t\t	  and change the data path.")
	print()

def PrintImageRange(title_, image_):
	minValue = np.min(image_)
	maxValue = np.max(image_)
	print(title_ + "(min, max) = (" + str(minValue) + ", " + str(maxValue) + ")")

def Check_DataAugmentation():
	canvas = np.zeros( [500, 1000, 3], dtype=np.uint8)

	for eachVideoPathFileName in LIST_OF_VIDEOS:
		batchOfImagesRGB = skvideo.io.vread(eachVideoPathFileName)
		batchOfAugmentedImages = DataAugmenter.Augment(batchOfImagesRGB)

		if batchOfImagesRGB.shape != batchOfAugmentedImages.shape:
			errorMessage = "Loaded image.shape = " + str(batchOfImagesRGB.shape) + "\n"
			errorMessage += "While augmentedImage.shape = " + str(augmentedImage.shape) + "\n"
			errorMessage += "Not Equal!"
			raise ValueError(errorMessage)

		w = batchOfImagesRGB.shape[2]
		h = batchOfImagesRGB.shape[1]

		i = 0
		while True:
			originalImage = batchOfImagesRGB[i]
			originalImage = cv2.cvtColor(originalImage, cv2.COLOR_RGB2BGR)
			canvas[:h, :w] = originalImage

			augmentedImage = batchOfAugmentedImages[i]
			augmentedImage = cv2.cvtColor(augmentedImage, cv2.COLOR_RGB2BGR)
			canvas[:h, 500:500+w] = augmentedImage
			
			cv2.imshow("Violence Detection", canvas)

			userResponse = cv2.waitKey(0)
			if userResponse == ord('n'):
				i += 1
				if i >= batchOfImagesRGB.shape[0]:
					break

			elif userResponse == ord('b'):
				i -= 1

			elif userResponse == ord('q'):
				raise StopIteration()



if __name__ == '__main__':
	if len(sys.argv) == 1:
		Check_DataAugmentation()


	else:
		PrintHelp()
