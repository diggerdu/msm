from data import Data
import numpy as np

from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn import mixture
from sklearn import svm

CLIENTR = 3000
TASKR = 2000



if __name__ == "__main__":
    data = Data()
    taskActive = list()
    taskDensity = list()
    price = list()
    label = list()
    for task in data.tasksCom:
        price.append(task.price)
        score = 0
        density = 0
        for client in data.clients:
            if task.haversine(client) < CLIENTR:
                score += np.log10(client.repu)
        taskActive.append(score)

        for other in data.tasksCom:
            if task.haversine(other) < TASKR:
                density += 1
        taskDensity.append(density)
        label.append(int(task.finished))


features = list(zip(taskActive, taskDensity))
features = np.array(features)
features[:, 0] = features[:, 0] / np.max(features[:, 0])
features[:, 1] = features[:, 0] / np.max(features[:, 1])

price = np.array(price)
priceMean = np.mean(price)
price = price - priceMean
priceMax = np.max(np.abs(price))
priceNormal = price / np.max(np.abs(price))



model = make_pipeline(PolynomialFeatures(2), LinearRegression(n_jobs=16))
model.fit(X=features, y=priceNormal)
print(model.predict(features[250:262]) * priceMax + priceMean)

print(priceNormal[250:262] * priceMax + priceMean)



print(features.shape)
print(priceNormal[:, np.newaxis].shape)
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
svc_cv = GridSearchCV(svc, tuned_parameters, cv=6,  scoring='average_precision', n_jobs=16)
poly = PolynomialFeatures(3)
svc_cv.fit(X=poly.fit_transform(gFeatures), y=label)


print(svc_cv.best_params_)

# print(Smodel.predict_proba(gFeatures))
# print(Smodel.predict(gFeatures))
S = svc_cv.predict(poly.fit_transform(gFeatures))
print(np.sum(S == np.array(label)))
