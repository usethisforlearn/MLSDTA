# MLSDTA
MLSDTA: Multimodal drug target binding affinity prediction using graph local substructure. This repository contains the source code and the data.


- 我们的数据文件在 https://drive.google.com/file/d/1ABjUhkMWNN0Z47nDn0Mk0vMlp7ANctqs/view?usp=drive_link 中，请先下载，然后解压

- 1、我们实验环境的依赖包都放在了requirements.txt文件中，同时，我们道出了enviroment.yml文件方便您创建虚拟环境
- 2、部署完成后，请使用python training_validation_Davis_KIBA.py训练模型
- 3、您可以使用python predict_with_pretrained_model_Davis_KIBA.py 利用保存的模型文件进行DTA预测，模型的预测结果将保存在 predict_result.csv 文件中
- 我们将davis数据集上训练好的模型文件上传到了https://drive.google.com/file/d/1kER88JYI8ZhwObv32V_8VJBctwR0kxx7/view?usp=drive_link
- 我们将kiba数据集上训练好的模型文件上传到了https://drive.google.com/file/d/1kI8ihfGguZP0OXUswgjB-gtwvmM4KaIw/view?usp=drive_link
- 您可以下载我们的模型文件，并通过 python predict_with_pretrained_model_Davis_KIBA.py 来复现我们的结果


~~~
conda create -n geometric python=3.7

conda activate geometric

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  rdkit

pip3 install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

conda install cudatoolkit

pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html

pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html

pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html

pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.1%2Bcu102.html

pip install torch-geometric
~~~

