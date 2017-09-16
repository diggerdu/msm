from data import Data
import numpy as np
import torch.optim as optim
from torch import Tensor
from networks import XUANet
from torch.autograd import Variable
import torch




def prepareData():
    data = Data()
    label = np.array([int(task.finished) for task in data.tasksCom])
    try:
        features = np.load("features.npy")
        print("###########features shape###########", features.shape)
        return features, label
    except:
        pass

    repuData = np.array([client.repu for client in data.clients])
    repuData = np.log10(repuData)
    repuDataMean = np.mean(repuData)
    repuData = repuData - repuDataMean
    repuData = repuData / np.max(np.abs(repuData))
    features = list()
    for task in data.tasksCom:
        dis = list()
        for client in data.clients:
            dis.append(task.haversine(client))
        dis = np.array(dis)[np.newaxis, :]
        curFeature = np.concatenate((dis, repuData[np.newaxis, :]), axis=0)[:, None, :]
        features.append(curFeature)

    features = np.array(features)

    dis = features[:, 0, :, :]
    disMean = np.mean(dis)
    dis = dis - disMean
    dis = dis / np.max(np.abs(dis))
    features[:, 0, :, :] = dis
    np.save('features', features)
    print("###########features shape###########", features.shape)
    return features, label


