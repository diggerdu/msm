import numpy as np
import matplotlib.pyplot as plt
from data import Data
from sklearn.cluster import KMeans
from sklearn.externals import joblib


data = Data()

TaskPos = np.array([[t.lon, t.lat] for t in data.tasksCom])
FinishedTaskPos = list(zip(*[[t.lon, t.lat] for t in data.tasksCom if t.finished]))
UnfinishedTaskPos = list(zip(*[[t.lon, t.lat] for t in data.tasksCom if not t.finished]))

kmModel = KMeans(n_clusters=2, random_state=1016, n_jobs=-1, max_iter=1e5).fit(TaskPos)
joblib.dump(kmModel, "models/kmModel.pkl")
y_pred = kmModel.predict(TaskPos)

np.save('plot', np.array(y_pred))



np.save('cluster', y_pred)
plt.subplot(211)
plt.scatter(*FinishedTaskPos, c='b', s=1)
plt.scatter(*UnfinishedTaskPos, c='r', s=1)

plt.subplot(212)
plt.scatter(TaskPos[:, 0], TaskPos[:, 1], c=y_pred, s=1)
plt.show()

