# ASLR-CNN
The purpose of this repository is to build off of the Hand Gesture Recognition models defined by the [Real Time Gesture Recognition](https://github.com/ahmetgunduz/Real-time-GesRec/tree/master) repository and alter their models for ASL. The project utilizes the ResNeXt models produced for the Jester hand gesture dataset and modifies them for the WLASL dataset. This dataset includes ~12,000 labeled videos of multiple signers covering 2,000 ASL words.

## Data Set
The dataset can be downloaded from the [WLASL kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data) using cURL (recommended) or the kaggle API. After downloading and unzipping the dataset the vidoes can be transformed with the frame_extractor.py script in the wlasl_data file. In it's default state the file will extract frames at 5fps, saving the jpg images in folders names for the video id, for example, "extracted\16704\frame_001.jpg". This script relies on ffmpeg commands so ensure it is installed and reachable when running the script.

```python frame_extractor.py <path_to_videos> <dest_folder> <frame_rate>```

## Dependencies
- python 3.12.0
- pyTorch
- torchvision
- matplotlib
- tensorboard
- conda
- jupyter notebook
- ffmpeg-python
- opencv-python
- scipy
- sickit-learn

## To launch
It is recommended to create a venv in the ALSR-CNN directory to install dependencies and proceed.  
To do this run the following commands from the ALSR-CNN directory
- ```python -m venv <environment_name>```
- ```source <path to venv>/Scripts/activate```
- Ensure the environment name appears in parenthesis in the command line after activation 
- ```pip install torch torchvision matplotlib tensorboard```
- ```deactivate``` (to make sure it shuts down properly)
