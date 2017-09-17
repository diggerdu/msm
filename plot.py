import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmaps
from data import Data
from sklearn.cluster import KMeans
from sklearn.externals import joblib


data = Data()

TaskPos = np.array([[t.lon, t.lat] for t in data.tasksCom])
# TaskPos = pd.DataFrame(data=TaskPos, columns=['lon', 'lat'])


ClientPos = np.array([[c.lon, c.lat] for c in data.clients if c.cluster==0])



MetroPos = np.array([[t.lon, t.lat] for t in data.stations])
FinishedTaskPos = list(zip(*[[t.lon, t.lat] for t in data.tasksCom if t.finished]))
UnfinishedTaskPos = list(zip(*[[t.lon, t.lat] for t in data.tasksCom if not t.finished]))


TaskPosDF = np.array([[t.lon, t.lat] for t in data.tasksCom if t.cluster==2])
TaskPosDF = pd.DataFrame(data=TaskPosDF, columns=['lon', 'lat'])

gmaps.scatter(TaskPosDF['lat'], TaskPosDF['lon'], colors='blue')


plt.subplot(211)
plt.scatter(*FinishedTaskPos, c='b', s=1)
plt.scatter(*UnfinishedTaskPos, c='r', s=1)
plt.scatter(MetroPos[:, 0], MetroPos[:, 1], c='g', s=1)

plt.subplot(212)
C = np.array(['b', 'g', 'k'])
TaskCluster = [t.cluster for t in data.tasksCom]
plt.scatter(TaskPos[:, 0], TaskPos[:, 1], c=C[TaskCluster], s=1)
plt.scatter(ClientPos[:, 0], ClientPos[:, 1], c='c', s=1)


plt.show()

