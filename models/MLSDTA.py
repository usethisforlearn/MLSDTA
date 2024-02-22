import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ASAPooling, GCNConv, SAGEConv, global_mean_pool as gap, global_max_pool as gmp

from models.pairnorm_layers import *


class MLSDTA(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=25, output_dim=128,
                 dropout_rate=0.1):
        super(MLSDTA, self).__init__()
        self.pooling_ratio = 0.5
        self.dropnode_rate = 0.2
        self.norm_mode = 'PN-SI'
        self.norm_scale = 1

        # drug graph feature extractor-------------------------------------------------------------------------------------------
        self.n_output = n_output

        self.gcn_drug1 = GCNConv(num_features_xd, num_features_xd)
        self.bn_gcn_drug1 = PairNorm(self.norm_mode, self.norm_scale)

        self.gcn_drug2 = GCNConv(num_features_xd, num_features_xd)
        self.bn_gcn_drug2 = PairNorm(self.norm_mode, self.norm_scale)

        self.gcn_drug3 = GCNConv(num_features_xd, num_features_xd)
        self.bn_gcn_drug3 = PairNorm(self.norm_mode, self.norm_scale)

        self.pool1 = ASAPooling(3 * num_features_xd, ratio=self.pooling_ratio, GNN=GCNConv)

        self.fc_druggraph = torch.nn.Linear(3 * num_features_xd, output_dim)  # 1024
        self.bn_fc_druggraph = nn.BatchNorm1d(output_dim)  # 1024

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # target sequence feature extractor -------------------------------------------------------------------------------------
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)

        self.conv_prot1 = nn.Conv1d(in_channels=1000, out_channels=4 * n_filters, kernel_size=3, padding=1)
        self.bn_conv_prot1 = nn.BatchNorm1d(4 * n_filters)
        self.conv_prot2 = nn.Conv1d(in_channels=4 * n_filters, out_channels=2 * n_filters, kernel_size=3, padding=1)
        self.bn_conv_prot2 = nn.BatchNorm1d(2 * n_filters)
        self.conv_prot3 = nn.Conv1d(in_channels=2 * n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn_conv_prot3 = nn.BatchNorm1d(n_filters)

        self.target_se = nn.Sequential(
            nn.AdaptiveAvgPool1d((1)),
            nn.Conv1d(n_filters, n_filters // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_filters // 16, n_filters, kernel_size=1),
            nn.Sigmoid()
        )

        self.fc_prot1 = nn.Linear(((1)*n_filters)*embed_dim, n_filters * embed_dim)
        self.bn_fc_prot1 = nn.BatchNorm1d(n_filters * embed_dim)
        self.fc_prot2 = nn.Linear(n_filters*embed_dim, output_dim)
        self.bn_fc_prot2 = nn.BatchNorm1d(output_dim)

        # drug sequence feature extractor-----------------------------------------------------------------------------------------------------
        self.embedding_xd = nn.Embedding(100, embed_dim)
        self.conv_smile1 = nn.Conv1d(in_channels=100, out_channels=2 * n_filters, kernel_size=3, padding=1)
        self.bn_conv_smile1 = nn.BatchNorm1d(2 * n_filters)
        self.conv_smile2 = nn.Conv1d(in_channels=2 * n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.bn_conv_smile2 = nn.BatchNorm1d(n_filters)
        # self.conv_smile3 = nn.Conv1d(in_channels=2 * n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        # self.bn_conv_smile3 = nn.BatchNorm1d(n_filters)

        self.drug_se = nn.Sequential(
            nn.AdaptiveAvgPool1d((1)),
            nn.Conv1d(n_filters, n_filters // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_filters // 16, n_filters, kernel_size=1),
            nn.Sigmoid()
        )

        self.fc_smile1 = nn.Linear(((1) * n_filters) * embed_dim, n_filters * embed_dim)
        self.bn_fc_smile1 = nn.BatchNorm1d(n_filters * embed_dim)
        self.fc_smile2 = nn.Linear(n_filters * embed_dim, output_dim)
        self.bn_fc_smile2 = nn.BatchNorm1d(output_dim)

        # target graph feature extractor----------------------------------------------------------------------------------------------------
        self.num_features_xt = 54
        self.gcn_target1 = GCNConv(self.num_features_xt, self.num_features_xt)
        self.bn_gcn_target1 = PairNorm(self.norm_mode, self.norm_scale)

        self.gcn_target2 = GCNConv(self.num_features_xt, self.num_features_xt)
        self.bn_gcn_target2 = PairNorm(self.norm_mode, self.norm_scale)

        self.gcn_target3 = GCNConv(self.num_features_xt, self.num_features_xt)
        self.bn_gcn_target3 = PairNorm(self.norm_mode, self.norm_scale)

        self.targetpool1 = ASAPooling(3 * self.num_features_xt, ratio=self.pooling_ratio, GNN=GCNConv)

        self.fc_targetgraph = torch.nn.Linear(3 * self.num_features_xt, output_dim)  # 1024
        self.bn_fc_targetgraph = nn.BatchNorm1d(output_dim)  # 1024


        # affinity predictor----------------------------------------------------------------------------------------------
        self.fc_concat1 = nn.Linear(8 * output_dim, 1024)
        self.bn_fc_concat1 = nn.BatchNorm1d(1024)
        self.fc_concat2 = nn.Linear(1024, 512)
        self.bn_fc_concat2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, DrugData, TargetData):
        # get drug graph data
        x, edge_index, batch = DrugData.x, DrugData.edge_index, DrugData.batch
        # get drug sequence data
        smiles = DrugData.smiles
        # get target graph data
        tx , target_edge_index ,tar_batch = TargetData.x, TargetData.edge_index ,TargetData.batch
        # get target sequence data
        target = TargetData.target  # batch_size=512, seq_len=1000

        # drug graph feature extraction----------------------------------------------------------------------------------------------------
        graph_xd = self.relu(self.bn_gcn_drug1(self.gcn_drug1(x, edge_index)))
        x1 = graph_xd
        graph_xd = self.relu(self.bn_gcn_drug2(self.gcn_drug2(graph_xd, edge_index)))
        x2 = graph_xd
        drug_mask = torch.rand(graph_xd.size(0)) > self.dropnode_rate
        drug_mask = drug_mask.to(x2.device)
        drug_mask = drug_mask.float().view(-1, 1)
        graph_xd = drug_mask * graph_xd
        graph_xd = self.relu(self.bn_gcn_drug3(self.gcn_drug3(graph_xd, edge_index)))
        x3 = graph_xd
        graph_xd = torch.cat([x1, x2, x3], dim=1)  # 16571*234
        graph_xd, edge_index, _, batch, _ = self.pool1(graph_xd, edge_index, None, batch)
        graph_xd = torch.add(gmp(graph_xd, batch), gap(graph_xd, batch)) / 2
        graph_xd = self.dropout(self.relu(self.bn_fc_druggraph(self.fc_druggraph(graph_xd))))

        # target sequence feature extraction------------------------------------------------------------------------------------------------------
        embedded_xt = self.embedding_xt(target)  # 512*1000*128
        conv_xt = self.relu(self.bn_conv_prot1(self.conv_prot1(embedded_xt)))  # 512*(32*4)*128
        conv_xt = self.relu(self.bn_conv_prot2(self.conv_prot2(conv_xt)))  # 512*(32*2)*128
        conv_xt = self.relu(self.bn_conv_prot3(self.conv_prot3(conv_xt)))  # 512*(32*1)*128
        # flatten
        conv_xt_se = self.target_se(conv_xt)
        conv_xt = conv_xt * conv_xt_se

        conv_xt = conv_xt.view(-1, 32*128)  # 512*1* (32*128)
        conv_xt = self.dropout(self.bn_fc_prot1(self.relu(self.fc_prot1(conv_xt))))
        conv_xt = self.dropout(self.bn_fc_prot2(self.relu(self.fc_prot2(conv_xt))))


        # drug sequence feature extraction
        embedded_xd = self.embedding_xd(smiles)  # 512*100*128
        conv_xd = self.relu(self.bn_conv_smile1(self.conv_smile1(embedded_xd)))  # 512*(64)*128
        conv_xd = self.relu(self.bn_conv_smile2(self.conv_smile2(conv_xd)))  # 512*(32)*128
        conv_xd_se = self.drug_se(conv_xd)
        conv_xd = conv_xd * conv_xd_se
        # flatten
        conv_xd = conv_xd.view(-1, 32 * 128)  # 512*1* (32*128)
        conv_xd = self.dropout(self.bn_fc_smile1(self.relu(self.fc_smile1(conv_xd))))
        conv_xd = self.dropout(self.bn_fc_smile2(self.relu(self.fc_smile2(conv_xd))))

        # target graph feature extraction
        graph_xt = self.relu(self.bn_gcn_target1(self.gcn_target1(tx, target_edge_index)))
        xt1 = graph_xt

        graph_xt = self.relu(self.bn_gcn_target2(self.gcn_target2(graph_xt, target_edge_index)))
        xt2 = graph_xt
        target_mask = torch.rand(graph_xt.size(0)) > self.dropnode_rate
        target_mask = target_mask.to(xt2.device)
        target_mask = target_mask.float().view(-1, 1)
        graph_xt = target_mask * graph_xt
        graph_xt = self.relu(self.bn_gcn_target3(self.gcn_target3(graph_xt, target_edge_index)))
        xt3 = graph_xt


        graph_xt = torch.cat([xt1, xt2, xt3], dim=1)  # 16571*234
        graph_xt, target_edge_index, _, tar_batch, _ = self.targetpool1(graph_xt, target_edge_index, None, tar_batch)
        graph_xt = torch.add(gmp(graph_xt, tar_batch), gap(graph_xt, tar_batch)) / 2
        graph_xt = self.dropout(self.relu(self.bn_fc_targetgraph(self.fc_targetgraph(graph_xt))))
        
        # multimodal feature attention
        atten_graph_xt = graph_xt * (F.softmax(graph_xt * conv_xt))
        atten_conv_xt = conv_xt * (F.softmax(graph_xt * conv_xt))
        atten_graph_xd = graph_xd * (F.softmax(graph_xd * conv_xd))
        atten_conv_xd = conv_xd * (F.softmax(graph_xd * conv_xd))
        
        # predictor

        xc = torch.cat([graph_xt, graph_xd,conv_xt, conv_xd,atten_graph_xt, atten_graph_xd,atten_conv_xt, atten_conv_xd], dim=1)
        xc = self.dropout(self.bn_fc_concat1(self.relu(self.fc_concat1(xc))))
        xc = self.dropout(self.bn_fc_concat2(self.relu(self.fc_concat2(xc))))
        out = self.out(xc)
        return out


