# utils_SELFIES.py
#
# This is a file from GraphDTA.
#
# Modified by: Shugang Zhang
# 修改于2022.10.14，将SMILES添加到data训练集里面
#

import os
import os.path as osp
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA

import torch
from lifelines.utils import concordance_index
import torch_geometric
from tqdm import tqdm

# InMemoryDataset  这个类是只存储一个图的，DataSet是存储多个图的
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, smile_tensor=None,target_graph=None,target_key=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.process(xd, xt, y, smile_graph, smile_tensor,target_graph,target_key)


    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_DrugData.pt', self.dataset + '_TargetData.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - 药物 smiles字符串
    # Y: 亲和力标签
    # xt - 蛋白质序列编码
    # smile_graph ： 药物图映射字典
    # smile_tensor ： 药物序列编码
    # target_graph : 蛋白质图映射字典
    # target_key ： 蛋白质key
    # Return: PyTorch-Geometric format processed data

    def process(self, xd, xt, y, smile_graph , smile_tensor, target_graph, target_key):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        Drug_data_list = []
        Target_data_list = []
        data_len = len(xd)
        print('data_len:',data_len)
        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            key = target_key[i]

            # convert SMILES to molecular representation using rdkit  ，这个地方的smile_graph 是一个字典，key是smiles，values是 对应的图
            c_size, features, edge_index = smile_graph[smiles]
            # 此处的 smile_tensor 不是函数，而是存储了映射关系的一个SMILES字典
            smile_ten = smile_tensor[smiles]


            tar_size, tar_features, tar_edge_index = target_graph[key]
            # 药物的图数据 和 序列数据
            DrugData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels])
                                )
            DrugData.smiles = torch.LongTensor([smile_ten])
            # DrugData.adj = torch.tensor(torch_geometric.utils.to_scipy_sparse_matrix(DrugData.edge_index).todense())
            DrugData.__setitem__('c_size', torch.LongTensor([c_size]))

            # 药物的图数据 和 序列数据
            TargetData = DATA.Data(
                                x=torch.Tensor([tar_features]).view(-1,54),
                                edge_index =torch.LongTensor(tar_edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels])
            )
            TargetData.target = torch.LongTensor([target])
            # TargetData.taradj = torch.tensor(torch_geometric.utils.to_scipy_sparse_matrix(TargetData.edge_index).todense())
            TargetData.__setitem__('tar_size', torch.LongTensor([tar_size]))

            # append graph, label and target sequence to data list
            Drug_data_list.append(DrugData)
            Target_data_list.append(TargetData)

        if self.pre_filter is not None:
            Drug_data_list = [data for data in Drug_data_list if self.pre_filter(data)]
            Target_data_list = [data for data in Target_data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            Drug_data_list = [self.pre_transform(data) for data in Drug_data_list]
            Target_data_list = [self.pre_transform(data) for data in Target_data_list]
        print('Graph construction done. Saving to file.')
        self.DrugData = Drug_data_list
        self.TargetData = Target_data_list


    def __len__(self):
        return len(self.DrugData)

    def __getitem__(self, idx):
        return self.DrugData[idx], self.TargetData[idx]

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 512
    LOG_INTERVAL = 512
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.SmoothL1Loss()

    for batch_idx, data in enumerate(train_loader):
        DrugData = data[0].to(device)
        TargetData = data[1].to(device)

        optimizer.zero_grad()
        output = model(DrugData, TargetData)
        # loss = loss_fn(output, DrugData.y.view(-1, 1).float().to(device))  + (1 - ci_tensor(output,  DrugData.y.view(-1, 1).float().to(device)).to(device))
        loss = loss_fn(output, DrugData.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB


def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0,1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci_tensor(f, y):  # p,y
    y = np.array(y.tolist())
    f = np.array(f.tolist())
    ind = np.argsort(y,axis=0)  # 对亲和力按从小到大排列
    ind = [x[0] for x in ind]
    y = [y[i][0] for i in ind]
    f = [f[i][0] for i in ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return torch.tensor(ci)


def ci(y, f):
    ind = np.argsort(y)  # 对亲和力按大小排序
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

# rm2 指标计算

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

