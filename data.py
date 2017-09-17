import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.externals import joblib
import numpy as np

Salary = [(18.3 * 1404.35 + 735 * 14.4) / (1404.35 + 735), 18.5, 14.4]
Cluster = ['GreatGuangzhou', 'Shenzhen', 'DongGuan']


def loadExcel(fn):
    xl = pd.ExcelFile(fn)
    dl = xl.parse(xl.sheet_names[0])
    return dl


def getSheet():
    data0 = loadExcel("A.xls")
    data1 = loadExcel("B.xlsx")
    data2 = loadExcel("C.xls")
    data3 = loadExcel("metro.xlsx")
    return data0, data1, data2, data3


class Object():
    def __init__(self, lat, lon, cluster):
        latMin = 22.4
        latMax = 23.5
        lonMin = 112.5
        lonMax = 114.50
        self.lat = lat
        self.lon = lon
        self.cluster = cluster
        self.normal = lat < latMax and lat > latMin and lon > lonMin and lon < lonMax
    def __repr__(self):
        return 'Object(gps=({}, {}), cluster:{})'.format(self.lat, self.lon)

    def haversine(self, other):
        lon1, lat1, lon2, lat2 = map(radians, [self.lon, self.lat, other.lon, other.lat])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r * 1000

class Client(Object):
    def __init__(self, gps, share, repu, cluster):
        Object.__init__(self, gps[0], gps[1], cluster)
        self.share = share
        self.repu = repu
    def __repr__(self):
        return 'Client(gps=({}, {}), share={}, repu={}, cluster:{})'.format(self.lat, self.lon , self.share, self.repu, self.cluster)

class TaskCom(Object):
    def __init__(self, lat, lon, price, finished, cluster):
        Object.__init__(self, lat, lon, cluster)
        self.price = price
        self.finished = finished
        self.cluster = cluster
    def __repr__(self):
        return 'TaskCompleted(gps:({}, {}), price:{}, finished:{}, cluster:{})'.format(self.lon,
                self.lat, self.price, self.finished, self.cluster)

class Data():
    def __init__(self):
        self.latMin = 22.4
        self.latMax = 23.5
        self.lonMin = 112.5
        self.lonMax = 114.50
        self.data0, self.data1, self.data2, self.data3 = getSheet()
        self.kmModel = joblib.load('models/kmModel.pkl')
        self.clients = list(filter(lambda o:o.normal, self.getClients()))
        self.tasksCom = list(filter(lambda o:o.normal, self.getTasks()))
        self.stations = list(filter(lambda o:o.normal, self.getMetro()))
        self.salary = Salary

    def getClients(self):
        gpsClients = self.data1['会员位置(GPS)'].tolist()
        gpsClients = list(map(lambda p:list(map(float, p.split(' ')[:2])), gpsClients))
        repuClients = list(map(float, self.data1['信誉值'].tolist()))
        shareClients = list(map(int, self.data1['预订任务限额'].tolist()))
        clusterClients = self.kmModel.predict(np.flip(np.array(gpsClients), axis=1))
        return list(map(lambda p:Client(*p),
                list(zip(gpsClients, shareClients, repuClients, clusterClients))))

    def getTasks(self):
        gpsTasksLat = list(map(float, self.data0['任务gps 纬度']))
        gpsTasksLon = list(map(float, self.data0['任务gps经度']))
        priceTask = list(map(float, self.data0['任务标价']))
        finishedTask = list(map(bool, self.data0['任务执行情况']))
        cluster = self.kmModel.predict(list(zip(gpsTasksLon, gpsTasksLat)))
        return list(map(lambda p:TaskCom(*p),
            list(zip(gpsTasksLat, gpsTasksLon, priceTask, finishedTask, cluster))))

    def getMetro(self):
        stationLon = list(map(float, self.data3['X轴坐标']))
        stationLat = list(map(float, self.data3['Y轴坐标']))
        stationCluster = self.kmModel.predict(list(zip(stationLon, stationLat)))
        return list(map(lambda p:Object(*p), list(zip(stationLat, stationLon, stationCluster))))





if __name__ == '__main__':
    data = Data()

    for c in data.clients:
        print(c)
