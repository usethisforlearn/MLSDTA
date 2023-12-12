import sys
import torch.nn as nn


from utils import *
from lifelines.utils import concordance_index
from create_data import create_dataset
import time

from models.MLSDTA import MLSDTA



datasets = ['davis', 'kiba']
modeling = [MLSDTA][0]



model_st = modeling.__name__

print("dataset:", datasets)
print("modeling:", modeling)

# determine the device in the following line
cuda_name = "cuda:1"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
# LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 2000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)

    start_time = time.time()
    train_data_file = "./data/" + dataset + "_traindata_"+ str(TRAIN_BATCH_SIZE) +".data"
    test_data_file = "./data/" + dataset + "_testdata_"+ str(TEST_BATCH_SIZE) +".data"
    if not (os.path.isfile(train_data_file) and os.path.isfile(test_data_file)):
        train_data, test_data = create_dataset(dataset)
        torch.save(train_data, train_data_file)  # 保存训练数据
        torch.save(test_data, test_data_file)  # 保存训练数据
    else:
        train_data = torch.load(train_data_file)
        test_data = torch.load(test_data_file)

    print('完成dataset加载。')
    # make data PyTorch mini-batch processing ready

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    print('完成dataloader加载。')
    end_time = time.time()
    all_time = end_time-start_time
    print('数据准备一共耗时:',all_time,'秒')
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
    result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch+1)
        G, P = predicting(model, device, test_loader)
        ret = [mse(G, P), concordance_index(G, P)]
        # 判断条件改为mse小于best_mse 且 ci大于 best_ci
        # if ret[0] < best_mse and ret[-1] > best_ci:
        # if epoch > 200 and best_mse > 0.21:
        #     print('200批次没有最佳效果，best_mse > 0.21，提前停止 ')
        #     break
        if ret[0] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch+1
            best_mse = ret[0]
            best_ci = ret[-1]
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
        elif (epoch - best_epoch)<500:
            print(ret[0], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
        else:
            print('提前停止 ''; best_mse,best_ci:', best_mse, best_ci,model_st, dataset)
            break
