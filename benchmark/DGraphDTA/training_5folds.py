import sys, os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from gnn import GNNNet
from utils import *
from emetrics import *
from data_process import create_dataset_for_5folds


datasets = [['davis', 'kiba'][1]]

cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][1]
print('cuda_name:', cuda_name)
fold = [0, 1, 2, 3, 4][0]
cross_validation_flag = True

TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LR = 0.001
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Main program: iterate over different datasets
result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = GNNNet()
model.to(device)
model_st = GNNNet.__name__
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


for dataset in datasets:
    train_data, valid_data = create_dataset_for_5folds(dataset, fold)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                               collate_fn=collate)

    best_mse = 1000
    best_test_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = 'models/model_' + model_st + '_' + dataset + '_' + str(fold) + '.model'

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)
        val = get_mse(G, P)
        cur_ci = get_ci(G,P)
        print('valid result:', val, best_mse)
        if val < best_mse and cur_ci > best_ci:
            best_mse = val
            best_ci = cur_ci
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print('rmse improved at epoch ', best_epoch, '; best_test_mse', best_mse, "; best_test_ci ",best_ci ,model_st, dataset, fold)
        elif epoch - best_epoch > 300:
            print('提前停止，best_epoch ', best_epoch, '; best_test_mse', best_mse, "; best_test_ci ",best_ci, model_st, dataset, fold)
            break
        else:
            print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, "; best_test_ci ",best_ci, model_st, dataset, fold)
