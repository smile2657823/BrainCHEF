from gtda.homology import VietorisRipsPersistence
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import pandas as pd
from gtda.time_series import TakensEmbedding


def PD2Embedding(hgDataset):
    hg_PD = []
    MAX_shape = []
    for hg in hgDataset:
        point_clouds = hg.H.to_dense().to('cpu').numpy()
        VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])  # Parameter explained in the text
        diagrams = VR.fit_transform([point_clouds])
        PD = diagrams[0]
        # one-hot
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot = one_hot_encoder.fit_transform(PD[:, 2].reshape(-1, 1))
        diff = np.argsort(PD[:, 0] - PD[:, 1])
        PD_one_hot = np.hstack((PD[diff][:,:2], one_hot[diff]))
        df = pd.DataFrame(PD_one_hot)
        # 对DataFrame按行去重，并保持顺序
        unique_df = df.drop_duplicates(keep='first')
        # 再转换回numpy数组
        unique_data = unique_df.values
        MAX_shape.append(unique_data.shape[0])
        hg_PD.append(unique_data)
    max_rows = max(MAX_shape)
    for i in range(len(hg_PD)):
        hg_PD[i] = np.pad(hg_PD[i], ((0, max_rows - hg_PD[i].shape[0]), (0, 0)), mode='constant', constant_values=0)
    hg_PD = torch.tensor(hg_PD)
    return hg_PD

def Time2Embedding(Time2TE):
    Time_PD = []
    MAX_shape = []
    for Time in Time2TE:
        point_clouds = Time
        VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])  # Parameter explained in the text
        diagrams = VR.fit_transform([point_clouds])
        PD = diagrams[0]
        # one-hot
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot = one_hot_encoder.fit_transform(PD[:, 2].reshape(-1, 1))
        diff = np.argsort(PD[:, 0] - PD[:, 1])
        PD_one_hot = np.hstack((PD[diff][:,:2], one_hot[diff]))
        df = pd.DataFrame(PD_one_hot)
        # 对DataFrame按行去重，并保持顺序
        unique_df = df.drop_duplicates(keep='first')
        # 再转换回numpy数组
        unique_data = unique_df.values
        MAX_shape.append(unique_data.shape[0])
        Time_PD.append(unique_data)
    max_rows = max(MAX_shape)
    for i in range(len(Time_PD)):
        Time_PD[i] = np.pad(Time_PD[i], ((0, max_rows - Time_PD[i].shape[0]), (0, 0)), mode='constant', constant_values=0)
    Time_PD = torch.tensor(Time_PD)
    return Time_PD

def time2PD(Time,time_delay=20,dimension=5):
    Time_PD = []
    row = []
    for i in range(len(Time)):
        print(i)
        TE = TakensEmbedding(time_delay=time_delay, dimension=dimension)
        Time2TE = TE.fit_transform(Time[i].T)
        PD = Time2Embedding(Time2TE)
        row.append(PD.shape[1])
        Time_PD.append(PD)   
    MIN_row = min(row)
    output_PD = []
    for j in range(len(Time_PD)):
        output_PD.append(Time_PD[j][:,:MIN_row,:])
    output_PD = torch.stack(output_PD, dim=0)
    return output_PD