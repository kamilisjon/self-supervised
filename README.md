# SSL_transfer_learning
## Setup up a virtual environment
1. Pull the repository
2. [Download Anaconda](https://docs.anaconda.com/anaconda/install/)
3. Follow steps in official TensorFlow website to setup virtual environment and install tensorflow-gpu
    1. For [Windows](https://www.tensorflow.org/install/pip#windows-native_1)
    2. For [Linux](https://www.tensorflow.org/install/pip#linux_1)
        1. Setup instructions in this README.md are more Windows friendly. To setup Linux system Windows instructions from step 4 should be carried out  
    3. Using environment name `tf_gpu` (`tf` is default in given instructions)
4. This repository is built using software versions defined below:
    1. tensorflow-2.10.* ([associated requirements](https://www.tensorflow.org/install/source#gpu:~:text=11.2-,tensorflow%2D2.10.0,-3.7%2D3.10))
    2. python3.8
    3. cuDNN8.1
    4. CUDA11.2

**IMPORTANT!** Make sure, that virtual environment is enabled, when further steps are carried out. Environment can be activated be running `conda activate ENVIRONMENT_NAME` in `Anaconda Prompt` (e.x. `conda activate tf_gpu`) 

5. Add virtual environment to Jupyter Notebook. In `Anaconda Prompt`:  ([source](https://towardsdatascience.com/setting-up-tensorflow-on-windows-gpu-492d1120414c))
    1. Run commands:
        1. `pip install jupyter ipykernel`
        2. `python -m ipykernel install --user --name ENVIRONMENT_NAME --display-name "KERNEL_DISPLAY_NAME"` (e.x. `python -m ipykernel install --user --name tf_gpu --display-name "tf_gpu"`)
    2. Run Jupyter Notebook and test if all good
        1. run: `jupyter notebook`
        2. Navigate to this repository
        3. Open `test_tf_gpu.ipynb`
        4. Select proper kernel. Navigate: Kernel -> Change kernel -> press KERNEL_DISPLAY_NAME (e.x. tf_gpu)
        5. Run the test cell
6. Install requirements. 
    1. In `Anaconda Prompt` navigate to this repository
    2. Run command: `pip install -r requirements.txt`

## Setup WandB
1. Add an API key of your WandB account to `.env`
2. Define a WandB project, that will be used to log training runs in `train.ipynb`

## Usage
Startup:
1. Open `Anaconda Prompt`and run commands:
    1. `conda activate ENVIRONMENT_NAME` (e.x. `conda activate tf_gpu`)
    2. `jupyter notebook`
2. Navigate to this repository
3. Preprocess data, train models, evaluate results


## Data preprocessing

## Model training 

## Model evaluation
