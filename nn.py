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
    repuDataMean = np.mean(repuData)
    repuData = repuData - repuDataMean
    features = list()
    for task in data.tasksCom:
        dis = list()
        for client in data.clients:
            dis.append(task.haversine(client))
        dis = np.array(dis)[np.newaxis, :]
        curFeature = np.concatenate((dis, repuData[np.newaxis, :]), axis=0)[:, None, :]
        features.append(curFeature)

    dis = features[:, 0, :, :]
    features[:, 1, :, :] =
    features = np.array(features)
    np.save('features', features)
    print("###########features shape###########", features.shape)
    return features, label


model = XUANet().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


features, label = prepareData()

BS = 16
for epoch in range(1, 100000):
    idx = np.random.choice(features.shape[0], BS)
    batchX = Variable(torch.cuda.FloatTensor(features[idx]))
    batchY = Variable(torch.cuda.LongTensor(label[idx].tolist()))
    _, loss = model(batchX, batchY)
    print(loss.data.cpu()[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #if epoch % 100 == 0:
    #    print('==>>> epoch: {}, train loss: {:.6f}'.format(epoch, loss))











'''
features = list(zip(taskActive, taskDensity))
features = np.array(features)
features[:, 0] = features[:, 0] / np.max(features[:, 0])
features[:, 1] = features[:, 0] / np.max(features[:, 1])

price = np.array(price)
priceMean = np.mean(price)
price = price - priceMean
priceMax = np.max(np.abs(price))
priceNormal = price / np.max(np.abs(price))

'''
