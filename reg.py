from data import Data
import numpy as np
import gmaps

from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn import mixture
from sklearn import svm
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

CLIENTR = 5000
TASKR = 2500
STATIONR = 10000



if __name__ == "__main__":
    data = Data()
    taskActive = list()
    taskDensity = list()
    taskTrans = list()
    price = list()
    label = list()
    for task in data.tasksCom:
        price.append(task.price)
        score = 0
        density = 0
        trans = 0
        for client in data.clients:
            if task.haversine(client) < CLIENTR:
                score += np.log10(client.repu)

        for other in data.tasksCom:
            if task.haversine(other) < TASKR:
                density += 1

        for station in data.stations:
            if task.haversine(station) < STATIONR:
                trans += 1

        taskActive.append(score)
        taskDensity.append(density)
        taskTrans.append(trans)
        label.append(int(task.finished))


features = list(zip(taskActive, taskDensity))

features = np.array(features)
print(features.shape)
features[:, 0] = features[:, 0] / np.max(features[:, 0])
features[:, 1] = features[:, 1] / np.max(features[:, 1])
# features[:, 2] = features[:, 2] / np.max(features[:, 2])

price = np.array(price)
priceMean = np.mean(price)
price = price - priceMean
priceMax = np.max(np.abs(price))
priceNormal = price / np.max(np.abs(price))



model = make_pipeline(PolynomialFeatures(2), LinearRegression(n_jobs=16))
model.fit(X=features, y=priceNormal)

pred = model.predict(features) * priceMax + priceMean
truth = priceNormal * priceMax + priceMean

idx = np.abs(pred - truth) < 4

features = features[idx]
priceNormal = priceNormal[idx]

model.fit(X=features, y=priceNormal)
pred = model.predict(features) * priceMax + priceMean
truth = priceNormal * priceMax + priceMean
print(r2_score(truth, pred))

idx = np.abs(pred - truth) < 2

features = features[idx]
priceNormal = priceNormal[idx]

model.fit(X=features, y=priceNormal)
pred = model.predict(features) * priceMax + priceMean
truth = priceNormal * priceMax + priceMean
print(r2_score(truth, pred))

x0= np.arange(0, 1, 0.001, dtype=np.float32)[:, np.newaxis]
x1 = 0.7 * np.ones((x0.shape))

y = model.predict(np.concatenate((x1, x0), axis=1))

plt.scatter(x0, y, c='c', s=1)
plt.show()





