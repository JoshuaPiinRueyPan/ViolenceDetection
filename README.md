# Violence Detection by CNN + LSTM

## Requirement
Python3

sk-video

scikit-image

TensorFlow 1.7.0

[imgaug](https://github.com/aleju/imgaug)
(This pakage has already contained in src/third_party)


## Quick Start
### Training
1. Download the fight/non-fight dataset from [here](http://visilab.etsii.uclm.es/personas/oscar/FightDetection/index.html)
   or, other fight/non-fight datasets is also supported if and only if you separate the fight and non-fight videos
   in the different directories.

2. To make the data catelogs that will tell the data manager where to load the videos, edit the file:
   tools/Train_Val_Test_spliter.py to specified the path to the dataset videos, the ratio to split the datasets into training,
   validation and test set.  And run such scripts, you will get three data catelogs: train.txt, val.txt, test.txt.

3. Edit the settings/DataSettings.py to specify where do you put the data catelogs:
```Shell
	PATH_TO_TRAIN_SET_CATELOG = 'MyPathToDataCatelog/train.txt'
	PATH_TO_VAL_SET_CATELOG = 'MyPathToDataCatelog/val.txt'
	PATH_TO_TEST_SET_CATELOG = 'MyPathToDataCatelog/test.txt'
```

4. Edit the settings/TrainSettings.py, and set the variables to fit your environment.  For example, you may want to edit
    the following variables:
```Shell
	MAX_TRAINING_EPOCH = 30

	EPOCHS_TO_START_SAVE_MODEL = 1
	PATH_TO_SAVE_MODEL = "MyPathToSaveTrainingResultsAndModels"
```

5. You're ready to train the model.  Type the following command to train:
```Shell
	python3 Train.py
```
or, if you set the Train.py to be executable, just type:
```Shell
	./Train.py
```

### Deploy
After you have trained a model, you can input a video and see its performance by following procedures:
1. Edit the settings/DeploySettings.py to set the variables to fit your environment.  For example, you may want to edit
   the following variables:
```Shell
	PATH_TO_MODEL_CHECKPOINTS = "PathToMyBestModelCheckpoint"
```
