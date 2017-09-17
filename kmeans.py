import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmaps
from data import Data
from sklearn.cluster import KMeans
from sklearn.externals import joblib


data = Data()

TaskPos = np.array([[t.lon, t.lat] for t in data.tasksCom])


kmModel = KMeans(n_clusters=3, random_state=1016, n_jobs=-1, max_iter=1e5).fit(TaskPos)
joblib.dump(kmModel, "models/kmModel.pkl")
y_pred = kmModel.predict(TaskPos)

C = np.array(['b', 'g', 'k'])
plt.scatter(TaskPos[:, 0], TaskPos[:, 1], c=C[y_pred], s=1)

plt.show()

