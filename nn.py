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
    repuData = np.log10(repuData + 10.) / np.log10(80)
    repuDataMean = np.mean(repuData)
    repuData = repuData - repuDataMean
    repuData = repuData / np.max(np.abs(repuData))
    features = list()
    for task in data.tasksCom:
        dis = list()
        for client in data.clients:
            dis.append(task.haversine(client))
        dis = np.array(dis)[np.newaxis, :]
        curFeature = np.concatenate((dis, repuData[np.newaxis, :], task.price * np.ones((1, 1877), dtype=np.float32)), axis=0)[:, None, :]
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


model = XUANet().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


features, label = prepareData()
evaX = Variable(torch.cuda.FloatTensor(features))
evaY = Variable(torch.cuda.FloatTensor(label.astype(float)))


BS = 64
Pidx = np.random.choice(features.shape[0], 6)
for epoch in range(1, 100000000000000000):
    idx = np.random.choice(features.shape[0], BS)
    # batchX = Variable(torch.cuda.FloatTensor(features[idx][:, 0, 0, :]))
    batchX = Variable(torch.cuda.FloatTensor(features[idx]))
    batchY = Variable(torch.cuda.FloatTensor(label[idx].astype(float)))
    _, loss = model(batchX, batchY)
    #    print(loss.data.cpu()[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prob, _ = model(evaX, evaY)
        prob = prob.data.cpu().numpy()
        print(prob[Pidx])
        pred = (prob > 0.5).astype(int)
        truth = evaY.data.cpu().numpy()
        print(pred[Pidx])
        print(truth[Pidx])
        print('accurancy:', np.sum(pred == truth) / features.shape[0])
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
