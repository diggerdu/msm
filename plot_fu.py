import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmaps
from data import Data
from sklearn.cluster import KMeans
from sklearn.externals import joblib


data = Data()




TaskPosDF = np.array([[t.lon, t.lat] for t in data.tasksFu])
TaskPosDF = pd.DataFrame(data=TaskPosDF, columns=['lon', 'lat'])

gmaps.scatter(TaskPosDF['lat'], TaskPosDF['lon'], colors='blue')


