# MLSDTA
MLSDTA: Multimodal drug target binding affinity prediction using graph local substructure. This repository contains the source code and the data.

### File
- **create_data.py**：Load data from Davis and KIBA.
- **training_validation_Davis_KIBA.py**:Train the model.
- **predict_with_pretrained_model_Davis_KIBA.py** : Use the existing model files to test the data in the test set. The predicted results will be saved in the **predict_result.csv** file.
- **requirements.txt**：Include packages for the virtual environment dependencies
- **utils.py**：Include functions for training, prediction, evaluation metrics, and more
- **models** ：Store the code for the model
- **baseline_predict.ipynb** : Load predictions of the baseline model and MLSDTA model, and generate charts.

### Docker
Considering that the Dockerfile is not related to the paper code, we upload the [Dockerfile](https://drive.google.com/file/d/1YZnhd3ATqFFlgeoDwg1XbERcUxTHZ6t5/view?usp=drive_link) to the cloud disk, and we also provide the [Dockerfile_CN](https://drive.google.com/file/d/1Nh84XlzRdZ0cH_MdGiV9tqJOSgzSZDjF/view?usp=drive_link) version, which uses Tsinghua University's image source when installing the dependency package with pip.
Here are the steps to use the dockerfile:
- Create a folder called MLSDTA
- Download the paper code package to MLSDTA and extract it
- Download the dataset file to MLSDTA and extract it
- Download the pre-trained model files model_MLSDTA_davis.model and model_MLSDTA_kiba.model to MLSDTA
- Generate an image with dockerfile: `sudo docker build -t image_mlsdta -f Dockerfile .`
- run the container: `sudo docker run -it image_mlsdta`

## How to train
 - Please download the data first, then unzip them, here is the link to download ([data](https://zenodo.org/records/10597707))
 - We provided bash files, and you can train the model with the bash command or python code provided below
 - Bash command:
~~~
bash training.sh
~~~
 - Python command：
~~~
python training_validation_Davis_KIBA.py
~~~

## How to test
 - Download the model parameter file. We uploaded the model file trained on the davis and kiba data set to model file ().
 - We have provided a bash file, and you can conduct model testing using the bash or Python commands provided below.
 - Bash command:
~~~
bash testing.sh
~~~
 - Python command:
~~~
python predict_with_pretrained_model_Davis_KIBA.py
~~~
After completing model testing, the predicted results will be saved in predict_result.csv. You can view the performance of the model in this file.



### Data
- Our data files are in [data](https://drive.google.com/file/d/1ABjUhkMWNN0Z47nDn0Mk0vMlp7ANctqs/view?usp=drive_link). Please download them first, and then unzip.
- We have placed the dependencies for our experimental environment in the **requirements.txt** file. Additionally, we have exported an **environment.yml** file for your convenience in creating a virtual environment.

### Train
- After deployment, please execute the following command to train the model.
~~~
python training_validation_Davis_KIBA.py
~~~
### Test
- You can execute the following command to utilize the saved model files for DTA prediction. The predicted results of the model will be saved in the **predict_result.csv** file.
~~~
python predict_with_pretrained_model_Davis_KIBA.py
~~~
- We have uploaded the trained model files on the Davis dataset to location [model_MLSDTA_davis.model](https://drive.google.com/file/d/1kER88JYI8ZhwObv32V_8VJBctwR0kxx7/view?usp=drive_link)
- We have uploaded the trained model files on the KIBA dataset to location [model_MLSDTA_kiba.model](https://drive.google.com/file/d/1kI8ihfGguZP0OXUswgjB-gtwvmM4KaIw/view?usp=drive_link)
- You can download our model files and reproduce our results by running 'python predict_with_pretrained_model_Davis_KIBA.py'.

### Baseline model
- [DeepDTA](https://github.com/hkmztrk/DeepDTA/)
- [ELECTRA-DTA](https://github.com/IILab-Resource/ELECTRA-DTA)
- [MATT_DTI](https://github.com/ZengYuni/MATT_DTI)
- [MFRDTA](https://github.com/JU-HuaY/MFR)
- [GraphDTA](https://github.com/thinng/GraphDTA)
- [DGraphDTA](https://github.com/595693085/DGraphDTA)
- [MGraphDTA](https://github.com/guaguabujianle/MGraphDTA)
~~~
conda create -n MLSDTA python=3.7
conda activate MLSDTA
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  rdkit
pip3 install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
conda install cudatoolkit
pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html
pip install torch-geometric==2.1.0
pip install lifelines==0.27.3
pip install networkx==2.6.3
~~~

