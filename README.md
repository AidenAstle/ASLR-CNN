# ASLR-CNN
The purpose of this repository is to build off of the Hand Gesture Recognition models defined by the [Real Time Gesture Recognition](https://github.com/ahmetgunduz/Real-time-GesRec/tree/master) repository and alter their models for ASL. The project utilizes the ResNeXt models produced for the Jester hand gesture dataset and modifies them for the WLASL dataset. This dataset includes ~12,000 labeled videos of multiple signers covering 2,000 ASL words.

## Data Set
The dataset can be downloaded from the [WLASL kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data) using cURL (recommended) or the kaggle API. After downloading and unzipping the dataset the vidoes can be transformed with the frame_extractor.py script in the wlasl_data file. In it's default state the file will extract frames at 5fps, saving the jpg images in folders names for the video id, for example, "extracted\16704\frame_001.jpg". This script relies on ffmpeg commands so ensure it is installed and reachable when running the script, follow the [installation instructions](https://ffmpeg.org/download.html) and ensure it's installed properly.

```python frame_extractor.py <path_to_videos> <dest_folder> <frame_rate>```

## Dependencies
- python 3.12.0
- conda
- Jupyter Notebook
- pyTorch
- torchvision
- matplotlib
- tensorboard
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
- ```pip install torch torchvision matplotlib tensorboard ffmpeg-python opencv-python scipy scikit-learn```
- ```deactivate``` (to make sure it shuts down properly)


## Models
As previosly stated, this project builds off of the hand recognition models produced by Okan Köpüklü, Ahmet Gunduz, Neslihan Kose, Gerhard Rigoll ([paper link](https://arxiv.org/abs/1901.10323)). This codebase contains scripts to train models based on thrit best performing ResNeXt 101 model. The main.py script builds and trains a new ResNeXt 101 model to classify the 2,000 hand signs labeled in the dataset, and the pretrained.py file performs a similar process, but loads and modifies the model weights of the best performing model from the original experiment, a ResNeXt 101 model which accepts RGB image data w/ a cardinality of 32. Other than the mdoel's themselves, the dataset file (wlasl.py) has been modified from the most similar jesterDS dataset from the original work to support a large and more varied selection of video inputs. Some utils have been altered and added as well to support the model training, including a custom collate function and tensorboard helper functions.