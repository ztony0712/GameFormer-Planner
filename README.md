
# GameFormer-Planner Simulation Pipeline
The fork is a simulation pipline construction of gameformer for arc researchers who supervised by Chengran. The trainning process has been finished and stored in ```training_log/Exp1/model_epoch_20_valADE_1.8124.pth```. The run_nuplan_test.py is set properly with defualt parameters. The fork would also be helpful to fix the issuses in nuboard

## How to start
### 1. Installation
Clone the simulation pipline by:
```
git clone https://github.com/ztony0712/GameFormer-Planner.git && cd GameFormer-Planner
```
The conda environment named 'gameformer' has been extracted to ```environment.yml``` file so that you can create it by running:
```
conda env create -f environment.yml
```

### 2. Run test program
Now you can skip **Data process** and **Training** cause the model file is ready, but the nuplan-devkit is still necessary. The default value is adaptable before running the test script:
```
python run_nuplan_test.py
```

If you need to modify anything instead of using our ready-to-use model file, do follow the official procedure! There would be no problem in **Data process**, but you need to manually split the train and validation dataset before **Training**. A script ```split_train_cal.py``` is written to solve this. Change the processed data path to yours, and you can also change the split ratio.

### 3. Repeat former experiences
- You don't need to run ```split_train_cal.py``` again if you want to check the results from the former experiences. You can input ```nuboard``` in terminal to start the visual interface, then upload the .nuboard file in the testing_log folder.

- If you can't initiate the nuboard, close every other nuboards in browser, stop corresponding programs in terminal and try again. Restart PC if those methods don't work.

- Clear the testing_log every time before you update this pipline cause the files are too big to upload.

### 4. Rendering video
[Install Chrome Browser and Chromedriver Ubuntu 20.04](https://skolo.online/documents/webscrapping/#pre-requisites). These tools are crucial to rendering video.

<font color="red" size="4">&#9888; IMPORTANT: The following part is the official readme! </font>
# GameFormer-Planner
This repository contains the code for the **innovation award** solution of the [nuPlan Planning Challenge](https://opendrivelab.com/AD23Challenge.html#Track4) at the CVPR'23 End-to-End Autonomous Driving Workshop. 

**GameFormer Planner: A Learning-enabled Interactive Prediction and Planning Framework for Autonomous Vehicles**
<br> Zhiyu Huang, Haochen Liu, Xiaoyu Mo, Chen Lv 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[Report]](https://opendrivelab.com/e2ead/AD23Challenge/Track_4_AID.pdf)**&nbsp;**[[GameFormer Paper]](https://arxiv.org/abs/2303.05760)**&nbsp;**[[Project Website]](https://mczhi.github.io/GameFormer/)**&nbsp;**[[Presentation]](https://www.youtube.com/watch?t=1204&v=ZwhXilQKULY&feature=youtu.be&ab_channel=OpenDriveLab)**

## Overview
This is an extension of [GameFormer](https://github.com/MCZhi/GameFormer), focusing on a comprehensive planning framework for autonomous driving. The framework consists of four key steps: feature processing, path planning, model query, and trajectory refinement. Comprehensive evaluations conducted on the nuPlan benchmark demonstrate the competitive performance of the proposed planning framework, validating its effectiveness in both open-loop and closed-loop tests.

![GameFormer Planner](https://github.com/MCZhi/GameFormer-Planner/assets/34206160/c36cb7f1-a5b3-4cef-84e7-8d8116485cbd)

## Getting started
### 1. Installation
To begin, please follow these steps:
- Download the [nuPlan dataset](https://www.nuscenes.org/nuplan#download) and set it up as described [here](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html). 
- Install the nuPlan devkit [here](https://nuplan-devkit.readthedocs.io/en/latest/installation.html) (version tested: v1.2.2). 
- Clone this repository and navigate into the folder:
```
git clone https://github.com/MCZhi/GameFormer-Planner.git && cd GameFormer-Planner
```
- Activate the environment created when installing the nuPlan-devkit:
```
conda activate nuplan
```
- Install the required packages:
```
pip install -r requirements.txt
```
- Add the following environment variable to your `~/.bashrc` (you can customize it):
```
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
```

### 2. Data process
Before training the GameFormer model, you need to preprocess the raw data using:
```
python data_process.py \
--data_path nuplan/dataset/nuplan-v1.1/splits/mini \
--map_path nuplan/dataset/maps \
--save_path nuplan/processed_data
```
Three arguments are necessary: ```--data_path``` to specify the path to the stored nuPlan dataset, ```--map_path``` to specify the path to the nuPlan map data, and ```--save_path``` to specify the path to save the processed data. 

Optional arguments like ```--scenarios_per_type``` and ```--total_scenarios``` can also be used to specify the amount of data to process.

### 3. Training
To train the GameFormer model, run:
```
python train_predictor.py \
--train_set nuplan/processed_data/train \
--valid_set nuplan/processed_data/valid
```
Two arguments are necessary: ```--train_set``` to specify the path to the processed training data and ```--valid_set``` to specify the path to the processed validation data.

Optional model arguments: ```--encoder_layers``` for the number of encoding layers, ```--decoder_layers``` for the number of interaction decoding layers, and ```--num_neighbors``` for the number of neighboring agents to predict (max number is 20).

Optional training parameters:```--train_epochs```, ```--batch_size```, and ```--learning_rate```.

### 4. Testing
To test the planning framework in nuPlan simulation scenarios, use:
```
python run_nuplan_test.py \
--experiment_name open_loop_boxes \
--data_path nuplan/dataset/nuplan-v1.1/splits/mini \
--map_path nuplan/dataset/maps \
--model_path training_log/your/model
```
Choose one of the three options ('open_loop_boxes', 'closed_loop_nonreactive_agents', 'closed_loop_reactive_agents') for ```--experiment_name```, and specify the ```--model_path```, which points to your trained model. Ensure to provide ```--data_path``` and ```--map_path``` arguments as done in the data process step.

Adjust the ```--scenarios_per_type``` and ```--total_scenarios``` arguments to control the number of scenarios tested.

**Make sure the model parameters in ```planner.py``` in ```_initialize_model``` match those used in training.**

## Contact
If you have any questions or suggestions, please feel free to open an issue or contact us (zhiyu001@e.ntu.edu.sg).

## Citation
If you find this repository useful for your research, please consider giving us a star &#127775; and citing our paper.

```angular2html
@InProceedings{Huang_2023_ICCV,
    author    = {Huang, Zhiyu and Liu, Haochen and Lv, Chen},
    title     = {GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3903-3913}
}
```
