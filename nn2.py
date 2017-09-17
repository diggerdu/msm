from data import Data
import numpy as np
import torch.optim as optim
from torch import Tensor
from networks import XUANet
from torch.autograd import Variable
import torch




def prepareData(cluster):
    data = Data()
    label = np.array([float(t.finished) for t in data.tasksCom if t.cluster==cluster])
    try:
        features = np.load("features.npy")
        print("###########features shape###########", features.shape)
        return features, label
    except:
        pass

    repuData = np.array([c.repu for c in data.clients if c.cluster==cluster])

    repuData = np.log10(repuData + 10.) / np.log10(80)
    repuDataMean = np.mean(repuData)
    repuData = repuData - repuDataMean
    repuData = repuData / np.max(np.abs(repuData))

    features = list()
    for task in [t for t in data.tasksCom if t.cluster==cluster]:
        dis = [task.haversine(c) for c in data.clients if c.cluster==cluster]
        dis = np.array(dis)[np.newaxis, :]
        s = dis.shape
        latFeatures = (task.lat - data.latMin) - (data.latMax - data.latMin)
        lonFeatures = (task.lon - data.lonMin) - (data.lonMax - data.lonMin)
        curFeature = np.concatenate((dis, repuData[np.newaxis, :]), axis=0)[:, None, :]
        features.append(curFeature)


    features = np.array(features)

    dis = features[:, 0, :, :]
    disMean = np.mean(dis)
    dis = dis - disMean
    dis = dis / np.max(np.abs(dis))
    features[:, 0, :, :] = dis
    #np.save('features', features)
    print("###########features shape###########", features.shape)
    return features, label

def save_networks(self, network, network_label):
    save_filename = '%net_%s.pth' % (network_label)
    save_path = os.path.join(self.save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    network.cuda()



cluster = 2
features, label = prepareData(cluster)
print(features.shape, label.shape)
inputLen = features.shape[-1]

model = XUANet(inputLen).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


evaX = Variable(torch.cuda.FloatTensor(features))
evaY = Variable(torch.cuda.FloatTensor(label))


BS = 64
Pidx = np.random.choice(features.shape[0], 6)
for epoch in range(1, 100000000000000000):
    idx = np.random.choice(features.shape[0], BS)
    # batchX = Variable(torch.cuda.FloatTensor(features[idx][:, 0, 0, :]))
    batchX = Variable(torch.cuda.FloatTensor(features[idx]))
    batchY = Variable(torch.cuda.FloatTensor(label[idx].astype(float)))
    _, loss = model(batchX, batchY)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print('loss', loss.data.cpu().numpy())

    if epoch % 200 == 0:
        prob, _ = model(evaX, evaY)
        prob = prob.data.cpu().numpy()
        pred = prob > 0.5
        truth = evaY.data.cpu().numpy() > 0.5

        prec = np.sum(truth==pred) / truth.shape[0]
        if prec > 0.8316:
            save_networks(model, 'cluster{}'.format(cluster))
        print(prec)
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
