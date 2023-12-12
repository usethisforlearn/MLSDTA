# MLSDTA
MLSDTA: Multimodal drug target binding affinity prediction using graph local substructure. This repository contains the source code and the data.

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

