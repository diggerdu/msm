import pandas as pd
from math import radians, cos, sin, asin, sqrt

def loadExcel(fn):
    xl = pd.ExcelFile(fn)
    dl = xl.parse(xl.sheet_names[0])
    return dl


def getSheet():
    data0 = loadExcel("A.xls")
    data1 = loadExcel("B.xlsx")
    data2 = loadExcel("C.xls")
    return data0, data1, data2


class Object():
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def haversine(self, other):
        lon1, lat1, lon2, lat2 = map(radians, [self.lon, self.lat, other.lon, other.lat])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r * 1000

class Client(Object):
    def __init__(self, gps, share, repu):
        Object.__init__(self, gps[0], gps[1])
        self.share = share
        self.repu = repu
    def __repr__(self):
        return 'Client(gps=({}, {}), share={}, repu={})'.format(self.lat, self.lon , self.share, self.repu)

class TaskCom(Object):
    def __init__(self, lat, lon, price, finished):
        Object.__init__(self, lat, lon)
        self.price = price
        self.finished = finished
    def __repr__(self):
        return 'TaskCompleted(gps=({}, {}), price={}, finished={})'.format(self.lat, self.lon, self.price, self.finished)

class Data():
    def __init__(self):
        self.data0, self.data1, self.data2 = getSheet()
        gpsClients = self.data1['会员位置(GPS)'].tolist()
        gpsClients = list(map(lambda p:list(map(float, p.split(' ')[:2])), gpsClients))
        repuClients = list(map(float, self.data1['信誉值'].tolist()))
        shareClients = list(map(int, self.data1['预订任务限额'].tolist()))
        self.clients = list(map(lambda p:Client(*p), list(zip(gpsClients, shareClients, repuClients))))
        self.tasksCom = self.getTasks()
    def getTasks(self):
        gpsTasksLat = list(map(float, self.data0['任务gps 纬度']))
        gpsTasksLon = list(map(float, self.data0['任务gps经度']))
        priceTask = list(map(float, self.data0['任务标价']))
        finishedTask = list(map(bool, self.data0['任务执行情况']))
        return list(map(lambda p:TaskCom(*p), list(zip(gpsTasksLat, gpsTasksLon, priceTask, finishedTask))))




if __name__ == '__main__':
    data = Data()
    print(data.clients[0])
    print(data.tasksCom[0])


