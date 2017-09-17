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

CLIENTR = 5000
TASKR = 2500
STATIONR = 500



if __name__ == "__main__":
    data = Data()
    taskActive = list()
    taskDensity = list()
    taskTrans = list()
    price = list()
    label = list()
    for task in [t for t in data.tasksCom if t.cluster==0]:
        price.append(task.price)
        score = 0
        density = 0
        trans = 0
        for client in [c for c in data.clients if c.cluster==0]:
            if task.haversine(client) < CLIENTR:
                score += np.log10(client.repu)

        for other in [t for t in data.tasksCom if t.cluster==0]:
            if task.haversine(other) < TASKR:
                density += 1

        #for station in data.stations:
        #    if task.haversine(station) < STATIONR:
        #        trans += 1

        taskActive.append(score)
        taskDensity.append(density)
        # taskTrans.append(trans)
        label.append(int(task.finished))


label = np.array(label)
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
label = label[idx]

model.fit(X=features, y=priceNormal)
pred = model.predict(features) * priceMax + priceMean
truth = priceNormal * priceMax + priceMean
print(r2_score(truth, pred))

idx = np.abs(pred - truth) < 2

features = features[idx]
priceNormal = priceNormal[idx]
label = label[idx]
print(priceNormal.shape)

model.fit(X=features, y=priceNormal)
pred = model.predict(features) * priceMax + priceMean
truth = priceNormal * priceMax + priceMean
print(r2_score(truth, pred))


np.save('pred', pred)
np.save('truth', truth)




gFeatures = np.concatenate((features, priceNormal[:,np.newaxis]), axis=1)
print(gFeatures.shape)


'''
label = np.array(label)
g = mixture.GaussianMixture(n_components=6, max_iter=100000)
g.fit(X=gFeatures, y=label)
P = g.predict(gFeatures)
print(np.sum(P == label))
'''

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 2*1e-3, 5*1e-3,  1e-4, 5*1e-4],
                     'C': [1, 5, 10, 20, 50, 80, 100, 200, 500, 1000]},
                    {'kernel': ['linear'], 'C': [1, 5, 10, 20, 50, 80, 100, 200, 500, 1000]}]

svc = svm.SVC(kernel='linear', C=1.0, probability=True, max_iter=-1)
svc_cv = GridSearchCV(svc, tuned_parameters, cv=8,  scoring='average_precision', n_jobs=16)
poly = PolynomialFeatures(3)
svc_cv.fit(X=poly.fit_transform(gFeatures), y=label)


print(svc_cv.best_params_)

S = svc_cv.predict(poly.fit_transform(gFeatures))
print(np.sum(S == np.array(label)))
