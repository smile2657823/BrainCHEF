from model.utils import LoggerRecord,Percentile
from model.func import train,validate,fun_error_img,fun_result
from model.models import MeanReadout,MlpAttention,SERO
from model.models import HGNN,HGNNP,JHGNN,HNHN,HyperGCN,HAN
from model.models import PDHGNN,PDHGNNP,PDJHGNN,PDHNHN,PDHyperGCN
from model.models.network import GAT, GCN, SAGE,CrossLevel
from model.models import UniGIN,UniSAGE,UniGAT,UniGCN
from sklearn.model_selection import StratifiedKFold
from model.structure import Hypergraph
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.models.pointnet_utils import TopologicalFeatureNetwork
from model.models.SelfGCN import PreModel
from scipy.sparse import coo_matrix 
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from model.structure import Graph
import numpy as np
import torch.optim as optim
import csv
import numpy as np
import scipy.io
from sklearn import metrics
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def getData(sourcedir='./data/ABIDE/',roi='aal90',k=20,hg_type='knn',time_delay=10,dimension=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    percentile = Percentile()
    filename = 'rois_'+ roi
    X = torch.load(os.path.join(sourcedir,'FC','{}.pth'.format(filename)))
    Y = pd.read_csv(os.path.join(sourcedir,'labels','participants.csv')).set_index('subject')['DX_GROUP'].values-1
    H = torch.load(os.path.join(sourcedir,'KNNHyperGraph', '{}_{}_H_knn_{}.pth'.format(hg_type,roi,str(k))))
    X_L = torch.load(os.path.join(sourcedir,'KNNLineGraph', '{}_{}_X_L_knn_{}.pth'.format(hg_type,roi,str(k))))
    A_L = torch.load(os.path.join(sourcedir,'KNNLineGraph', '{}_{}_A_L_knn_{}.pth'.format(hg_type,roi,str(k))))
    PD_time = torch.load(os.path.join(sourcedir, 'PD','PD_time_{}_time_delay_{}_dimension_{}.pth'.format(roi,str(time_delay),str(dimension))))
    A = torch.load(os.path.join(sourcedir, 'KNNHyperGraph', '{}_{}_A_knn_{}.pth'.format(hg_type,roi,str(k))))
    # PD_time = PD_time.permute(0,1,3,2).float()
    # PD_time = PD_time.reshape(PD_time.shape[0], PD_time.shape[1], -1)
    # PD_time = PD_time.permute(0, 2, 1).reshape(871, 5, -1)
    hgDataset = []
    for i in range(len(H)):
        hyperedges = []
        for j in range(H[i].shape[1]):
            vertex_indices = torch.where(H[i][:, j] == 1)[0]
            vertex_indices = tuple(index.item() for index in vertex_indices)
            hyperedges.append(vertex_indices)
        hg = Hypergraph(H[0].shape[0],hyperedges)
        hgDataset.append(hg)
    A_L_edge_indexs = []
    A_L_edge_weights = []
    H_edge_indexs = []
    H_edge_weights = []
    A_edge_indexs = []
    A_edge_weights = []
    for index in range(len(A_L)):
        # linegraph
        row1, col1 = A_L[index].nonzero(as_tuple=True)  
        edge_weight1 = A_L[index][row1, col1]
        # 提取非零元素的权重
        edge_index1 = torch.stack([row1, col1], dim=0)
        A_L_edge_indexs.append(edge_index1)
        A_L_edge_weights.append(edge_weight1.float())
        # hypergraph:H
        row2, col2 = H[index].nonzero(as_tuple=True)  
        edge_weight2 = H[index][row2, col2]
        # 提取非零元素的权重
        edge_index2 = torch.stack([row2, col2], dim=0)
        H_edge_indexs.append(edge_index2)
        H_edge_weights.append(edge_weight2.float())
        # hypergraph:A
        row3, col3 = A[index].nonzero(as_tuple=True)  
        edge_weight3 = A[index][row3, col3]
        # 提取非零元素的权重
        edge_index3 = torch.stack([row3, col3], dim=0)
        A_edge_indexs.append(edge_index3)
        A_edge_weights.append(edge_weight3.float())
    return torch.tensor(Y), hgDataset,X,X_L,A_L_edge_indexs,A_L_edge_weights,H_edge_indexs,H_edge_weights,A_edge_indexs,A_edge_weights,PD_time.float(),H,A
class DHGDataset(Dataset):
    def __init__(self, indexList):
        self.indexList = indexList

    def __len__(self):
        return len(self.indexList)

    def __getitem__(self, index):
        # 假设每个元组包含输入数据和目标数据
        indexs, X,PD_time,labels = self.indexList[index]
        #         print(indexs)
        return indexs, X,PD_time,labels
class Model(nn.Module):
    def __init__(self,
                 hgDataset,
                 H,
                 A,
                 X_L,
                 A_L_edge_indexs,
                 A_L_edge_weights,
                 H_edge_indexs,
                 H_edge_weights,
                 A_edge_indexs,
                 A_edge_weights,
                 hypergraph_type='HGNN',
                 linegraph_type = 'GCN',
                 HGNN_layer = 2,
                 GNN_layer = 2,
                 hidden_dims=[200,200],
                 dropout = 0.2,
                 mask_rate = 0.3,
                 topo = False,
                 linegraph = False,
                 sup = False,
                 CL = False
                ):
        super(Model, self).__init__()
        self.hgDataset = hgDataset
        self.H = H
        self.A = A
        self.X_L = X_L
        self.A_L_edge_indexs = A_L_edge_indexs
        self.A_L_edge_weights = A_L_edge_weights
        self.H_edge_indexs = H_edge_indexs
        self.H_edge_weights = H_edge_weights
        self.A_edge_indexs = A_edge_indexs
        self.A_edge_weights = A_edge_weights
        self.hypergraph_type = hypergraph_type
        self.linegraph_type = linegraph_type
        self.HGNN_layer = HGNN_layer
        self.GNN_layer = GNN_layer
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.mask_rate = mask_rate
        self.topo = topo
        self.linegraph = linegraph
        self.sup = sup
        self.CL = CL
        self.nodes = self.hgDataset[0].num_v
        # print(self.dropout)
        if hypergraph_type=='HGNN':
            self.HGNN = HGNN(self.nodes, self.hidden_dims[1], HGNN_layer,self.dropout)
        if hypergraph_type=='HGNNP':
            self.HGNN = HGNNP(self.nodes, self.hidden_dims[1], HGNN_layer,self.dropout)
        if hypergraph_type=='JHGNN':
            self.HGNN = JHGNN(self.nodes, self.hidden_dims[1], HGNN_layer,self.dropout)
        if hypergraph_type=='HNHN':
            self.HGNN = HNHN(self.nodes, self.hidden_dims[1], HGNN_layer,self.dropout)
        if hypergraph_type=='HyperGCN':
            self.HGNN = HyperGCN(self.nodes, self.hidden_dims[1], HGNN_layer,self.dropout)
        if hypergraph_type=='HAN':
            self.HGNN = HAN(self.nodes, self.hidden_dims[1], HGNN_layer,self.dropout)
            # self.HGNN2 = HypergraphAttentionNetwork(self.nodes, self.hidden_dims[1],self.dropout)
        if linegraph_type == 'GCN':
            if self.sup:
                self.LineGNN = PreModel(self.nodes, self.hidden_dims[0],self.hidden_dims[0], GNN_layer,self.dropout,mask_rate=self.mask_rate)
            else:
                self.LineGNN = GCN(self.nodes, self.hidden_dims[0],self.hidden_dims[0], GNN_layer,self.dropout)
        self.Readout = SERO(self.hidden_dims[0])
        self.Topo = TopologicalFeatureNetwork(5,self.nodes)
        if self.linegraph:
            if self.CL:
                # self.W1 = nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
                # self.W2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
                # self.AR = nn.Linear(self.hidden_dims[0] + self.hidden_dims[1],1)
                # print(self.AR)
                self.AR = CrossLevel(self.hidden_dims[0])
            if self.topo:
                self.FC = nn.Linear(self.hidden_dims[0]*3,2)
            else:
                self.FC = nn.Linear(self.hidden_dims[0] + self.hidden_dims[1],2)
        else:
            self.FC = nn.Linear(self.hidden_dims[0],2)
    def forward(self, indexs, X,PD_time,device):
        if self.linegraph:
            outputs = []
            Line_outputs = []
            PD_outputs = []
            for i in range(len(indexs)):
                if self.sup:
                    Line_output,loss = self.LineGNN(self.X_L[indexs[i]].to(device),self.A_L_edge_indexs[indexs[i]].to(device),self.A_L_edge_weights[indexs[i]].to(device))
                else:
                    Line_output = self.LineGNN(self.X_L[indexs[i]].to(device),self.A_L_edge_indexs[indexs[i]].to(device),self.A_L_edge_weights[indexs[i]].to(device))
                if self.hypergraph_type == 'HAN':
                    hg = self.hgDataset[indexs[i]].to(device)
                    H_edge_index = self.H_edge_indexs[indexs[i]].to(device)
                    H_edge_weight = self.H_edge_weights[indexs[i]].to(device)
                    A_edge_index = self.A_edge_indexs[indexs[i]].to(device)
                    A_edge_weight = self.A_edge_weights[indexs[i]].to(device)
                    H = self.H[indexs[i]].to(device)
                    A = self.A[indexs[i]].to(device)
                    X_L = self.X_L[indexs[i]].to(device)
                    # if self.topo:
                    #     # print(PD_time[i].shape)
                    #     x = X[i] + self.Topo(PD_time[i])
                    #     output = self.HGNN(x,hg,H_edge_index,H_edge_weight,A_edge_index,A_edge_weight,H,A,X_L,aggr = "softmax_then_sum")
                    # else:
                    #     output = self.HGNN(X[i],hg,H_edge_index,H_edge_weight,A_edge_index,A_edge_weight,H,A,X_L,aggr = "softmax_then_sum")
                    output = self.HGNN(X[i],hg,H_edge_index,H_edge_weight,A_edge_index,A_edge_weight,H,A,X_L,aggr = "softmax_then_sum")
                else:
                    output = self.HGNN(X[i], self.hgDataset[indexs[i]].to(device))
                if self.CL:
                    H_edge_index = self.H_edge_indexs[indexs[i]].to(device)
                    H = self.H[indexs[i]].to(device)
                    Line_output = self.AR(Line_output,output,H_edge_index,H)
                PD_outputs.append(self.Topo(PD_time[i]))
                Line_output = self.Readout(Line_output.unsqueeze(0))
                Line_outputs.append(Line_output[0].squeeze(0))
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)
            Line_outputs = torch.stack(Line_outputs, dim=0)
            PD_outputs = torch.stack(PD_outputs, dim=0)
            outputs = self.Readout(outputs)
            outputs= torch.cat((outputs[0], Line_outputs), dim=1)
            if self.topo:
                topos = self.Readout(PD_outputs)[0]
                # print(outputs.shape)
                # print(topos.shape)
                outputs= torch.cat((outputs, topos), dim=1)
                outputs = self.FC(outputs)
            else:
                outputs = self.FC(outputs)
        else:
            outputs = []
            for i in range(len(indexs)):
                if self.hypergraph_type == 'HAN':
                    hg = self.hgDataset[indexs[i]].to(device)
                    H_edge_index = self.H_edge_indexs[indexs[i]].to(device)
                    H_edge_weight = self.H_edge_weights[indexs[i]].to(device)
                    A_edge_index = self.A_edge_indexs[indexs[i]].to(device)
                    A_edge_weight = self.A_edge_weights[indexs[i]].to(device)
                    H = self.H[indexs[i]].to(device)
                    A = self.A[indexs[i]].to(device)
                    X_L = self.X_L[indexs[i]].to(device)
                    output = self.HGNN(X[i],hg,H_edge_index,H_edge_weight,A_edge_index,A_edge_weight,H,A,X_L,aggr = "softmax_then_sum")
                else:
                    output = self.HGNN(X[i], self.hgDataset[indexs[i]].to(device))
                outputs.append(output)
            outputs = torch.stack(outputs, dim=0)
            # print(outputs.shape)
            outputs = self.Readout(outputs)
            # print(outputs[0].shape)
            outputs = self.FC(outputs[0])
        if self.sup:
            return outputs,loss
        else:
            return outputs
def main(file_path,
         epochs=100,
         lr=0.00005,
         num_folds=5,
         num_classes=2,
         roi='aal90',
         alpha=1,
         k = 20,
         hypergraph_type='HGNN',
         linegraph_type = 'GCN',
         HGNN_layer = 2,
         GNN_layer = 2,
         hidden_dims=[90,90],
         dropout = 0.2,
         mask_rate =0.3,
         topo = False,
         linegraph = False,
         sup = False,
         CL = False
        ):
    Y,hgDataset,X,X_L,A_L_edge_indexs,A_L_edge_weights,H_edge_indexs,H_edge_weights,A_edge_indexs,A_edge_weights,PD_time,H,A =getData(roi=roi,k=k)
    print("超图模型：", hypergraph_type, "线图模型：",linegraph_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cc200
    #     k_fold = StratifiedKFold(num_folds,shuffle=True,random_state = 403)
    # aal116
    k_fold = StratifiedKFold(num_folds, shuffle=True, random_state=403)
    file_name = os.path.join(file_path, 'result.txt')
    z = [i for i in range(len(Y))]
    for fold in range(num_folds):
        logger = LoggerRecord(num_folds, num_classes)
        train_idx, test_idx = list(k_fold.split(z, Y))[fold]
        error = test_idx
        train_idx = [
            (train_idx[i], X[train_idx[i]],PD_time[train_idx[i]], Y[train_idx[i]])
            for i in range(len(train_idx))
        ]
        test_idx = [(test_idx[i], X[test_idx[i]], PD_time[test_idx[i]], Y[test_idx[i]])
                    for i in range(len(test_idx))]
        trainDataset = DHGDataset(train_idx)
        testDataset = DHGDataset(test_idx)
        trainDataloader = DataLoader(trainDataset, batch_size=32, shuffle=True)
        testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False)
        model = Model(hgDataset,
                      H,
                      A,
                     X_L,
                     A_L_edge_indexs,
                     A_L_edge_weights,
                     H_edge_indexs,
                     H_edge_weights,
                     A_edge_indexs,
                     A_edge_weights,
                     hypergraph_type,
                     linegraph_type,
                     HGNN_layer,
                     GNN_layer,
                     hidden_dims,
                     dropout,
                     mask_rate,
                     topo,
                     linegraph,
                     sup,
                     CL)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        # criterion1 = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.0005)
        #         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        best_acc = 0.0
        best_auc = 0.0
        for epoch in range(epochs):
            train_metric, train_loss = train(model,
                                             trainDataloader,
                                             optimizer,
                                             criterion,
                                             fold,
                                             logger,
                                             device,
                                             sup,
                                             alpha=alpha)
            logger.initialize(fold)
            test_metric, test_loss, error_index = validate(model,
                                                           testDataloader,
                                                           criterion,
                                                           fold,
                                                           logger,
                                                           device,
                                                           sup,
                                                           alpha=alpha)
            # 保存错误分类的图片
            error_list = [(i, error[i], error_index[i])
                          for i in range(len(error))]
            # 创建错误分类图片保存的路径
            if not os.path.exists(file_path + "/error/fold " + str(fold) +
                                  "/"):
                os.makedirs(file_path + "/error/fold " + str(fold) + "/")
            # 打开文件，指定写入模式
            with open(
                    file_path + "/error/fold " + str(fold) + "/epoch " +
                    str(epoch) + '.txt', 'w') as file:
                # 遍历列表并将每个元素写入文件
                for item in error_list:
                    file.write(str(item) + '\n')
            print(
                f"Fold:{fold+1} Epoch: {epoch+1} Train Loss: {train_loss:.4f} Val Loss: {test_loss:.4f} Train Accuracy: {train_metric['accuracy']:.4f} Val Accuracy: {test_metric['accuracy']:.4f}"
            )
            if test_metric['accuracy'] > best_acc:
                best_acc = test_metric['accuracy']
                print("=====Best-ACC:{:.3f}=====".format(best_acc))
                torch.save(
                    model.state_dict(),
                    os.path.join(file_path,
                                 'fold_{}_bestacc_model.pth'.format(fold + 1)))
            if test_metric['roc_auc'] > best_auc:
                best_auc = test_metric['roc_auc']
                print("=====Best-AUC:{:.3f}=====".format(best_auc))
                torch.save(
                    model.state_dict(),
                    os.path.join(file_path,
                                 'fold_{}_bestauc_model.pth'.format(fold + 1)))
            logger.record(train_metric, test_metric, train_loss, test_loss,
                          epoch, fold, file_name)
        torch.cuda.empty_cache()
def fileSave(epochs=100,
             lr=0.005,
             num_folds=5,
             num_classes=2,
             roi='aal90',
             alpha=1,
             k=20,
             hypergraph_type='HGNN',
             linegraph_type = 'GCN',
             HGNN_layer = 2,
             GNN_layer = 2,
             hidden_dims=[200,200],
             dropout = 0.2,
             mask_rate=0.3,
             topo = False,
             linegraph = False,
             sup = False,
             CL = False
            ):
    if linegraph:
        file_Path = './result_' +roi+ '/' + hypergraph_type+'_'+linegraph_type+'_'+str(k)+ '/' + 'parameter'
        if sup:
            file_Path = './result_' +roi+ '/' + hypergraph_type+'_'+linegraph_type+'_'+str(k)+'_'+'sup' + '/' + 'parameter/'
            if CL:
                file_Path = './result_' +roi+ '/' + hypergraph_type+'_'+linegraph_type+'_'+str(k)+'_'+'sup'+'_'+'CL' +"_"+str(mask_rate)+ '/' + 'parameter/'
        else:
            if CL:
                file_Path = './result_' +roi+ '/' + hypergraph_type+'_'+linegraph_type+'_'+str(k)+'_'+'CL' + '/' + 'parameter/'
    if not os.path.exists(file_Path):
        os.makedirs(file_Path)
    main(file_path=file_Path,
         epochs = epochs,
         lr = lr,
         num_folds = num_folds,
         num_classes = num_classes,
         roi = roi,
         alpha = alpha,
         k = k,
         hypergraph_type = hypergraph_type,
         linegraph_type = linegraph_type,
         HGNN_layer = HGNN_layer,
         GNN_layer = GNN_layer,
         hidden_dims = hidden_dims,
         dropout = dropout,
         mask_rate=mask_rate,
         topo = topo,
         linegraph = linegraph,
         sup = sup,
         CL = CL)
    result_path = file_Path + '/result.txt'
    if linegraph:
        save_path = './result_' +roi+ '/' + hypergraph_type+'_'+linegraph_type +'_'+str(k)+ '/' + 'figure/'
        if sup:
            save_path = './result_' +roi+ '/' + hypergraph_type+'_'+linegraph_type+'_'+str(k)+'_'+'sup' + '/' + 'figure/'
            if CL:
                save_path = './result_' +roi+ '/' + hypergraph_type+'_'+linegraph_type+'_'+str(k)+'_'+'sup'+'_'+'CL' +"_"+str(mask_rate)+ '/' + 'figure/'
        else:
            if CL:
                save_path = './result_' +roi+ '/' + hypergraph_type+'_'+linegraph_type+'_'+str(k)+'_'+'CL' + '/' + 'figure/'
        # if topo:
        #     save_path = './result_' +roi+ '/' + hypergraph_type+'_'+linegraph_type+'_'+str(k)+'_'+'topo' + '/' + 'figure/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fun_result(result_path, save_path, fold_num=num_folds)