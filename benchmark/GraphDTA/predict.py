import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import time


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


# datasets = ['davis','kiba'][0] 
# datasets = ['kiba']
datasets = ['davis']

modeling = [GCNNet][0]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3])) 
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000
result = []
print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        
        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)

        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        
        if os.path.isfile(model_file_name):
            model.load_state_dict(torch.load(model_file_name), strict=False)


            s = time.time()
            G,P = predicting(model, device, test_loader)
            with open(model_st+'_predict_value_'+dataset+'.csv', 'w') as f:
                f.write('true_label,predict_value\n')
                for i in range(len(G)):
                    f.write(str(G[i])+', '+str(P[i]) + '\n')
            e = time.time()
            a = e - s
            print('模型'+model_st+'预测一共耗时:', a, '秒')
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P),rm2(G,P)]
            ret =[dataset, model_st] + [round(e, 3) for e in ret]
            result += [ ret ]
            print('dataset,model,rmse,mse,pearson,spearman,ci,rm2')
            print(ret)
        else:
            print('model is not available!')

#--------------------------------------------------------------------------------------------------------------------------------
#     for modeling in modelings:
        
#         model_st = modeling.__name__
#         print('\npredicting for ', dataset, ' using ', model_st)
#         # training the model
#         device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
#         model = modeling().to(device)

#         model_file_name = 'model_' + model_st + '_' + dataset + '.model'
#         if os.path.isfile(model_file_name):
#             model.load_state_dict(torch.load(model_file_name), strict=False)
#             s = time.time()
#             G,P = predicting(model, device, test_loader)
#             with open(model_st+'_predict_value_'+dataset+'.csv', 'w') as f:
#                 f.write('true_label,predict_value\n')
#                 for i in range(len(G)):
#                     f.write(str(G[i])+', '+str(P[i]) + '\n')
#             e = time.time()
#             a = e - s
#             print('模型'+model_st+'预测一共耗时:', a, '秒')
#             ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P),rm2(G,P)]
#             ret =[dataset, model_st] + [round(e, 3) for e in ret]
#             result += [ ret ]
#             print('dataset,model,rmse,mse,pearson,spearman,ci,rm2')
#             print(ret)


#         else:
#             print('model is not available!')
                