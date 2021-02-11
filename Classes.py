import numpy as np 
import math
import pm4py as pm
import random
import csv
import statistics as st
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from datetime import timedelta,datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os as os
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute
from sklearn import cluster as skcl
import sklearn as sk 
from sklearn.neighbors import NearestNeighbors,BallTree
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
import xes
from copy import deepcopy
import time 

pd.set_option("display.max_rows", None, "display.max_columns", None) 

def importLog (path,name):
    log = pm.read_xes(os.path.join(path,name+".xes"))
    return log


def getAttributes (event, attributes):
    al = {}
    for a in attributes:
        try :
            al[a] = Attribute(a,event[a])
        except:
            al[a] = Attribute(a,None)
    return al

def returnState (e, where="lifecycle"):
    if where == "lifecycle" and "lifecycle:transition" in e:
        return e["lifecycle:transition"]
    elif where=="Activity" and "START" in e["Activity"]:
        return "start"
    else:
        return "complete"

def findDistinct (session):
    event_list = []
    for e in session:
        if e.name not in event_list:
            event_list.append(e.name)
    return event_list

def concatName (events,ind):
    name = ""
    for i,n in enumerate(ind):
        name += events[n]
        if i != len(ind)-1:
            name += " & "
    return name

def normalize(encoded,limit=-1,factor = 1):
    if limit != -1:
        esum = sum(encoded[0:limit])*factor
    else:
        esum = sum(encoded)*factor
    for i,e in enumerate(encoded[0:limit]):
        if e != 0 :
            encoded[i] = e/esum
    return encoded

def normalizeAttr (attr):
    asum = np.sum(attr,axis=0)
    return list(np.divide(attr,asum))

def distance (center, enc):
    difference = np.diff([center,enc], axis = 0)
    return np.sqrt(np.sum(np.power(difference,2)))

def allDistances (centers, enc, y_pred):
    alldist = []
    for i,e in enumerate(enc):
        if y_pred[i] == -1:
            for y in range(max(y_pred)+1) :
                dict1 = {"from": y, "dist": distance(centers[y],e) , "session":i}
                alldist.append(dict1)
    return pd.DataFrame(alldist,columns= ["from","dist","session"])

def assignNoisyPoints(y_pred,enc,centers):
    print("Assigned noisy points to closest cluster")
    alldist = allDistances(centers, enc, y_pred)
    for i in range(len(enc)) :
        if y_pred[i] == -1:
            allMe = alldist[alldist["session"]==i]
            minInd = allMe[["dist"]].idxmin()
            y_pred[i] = alldist.at[int(minInd),"from"]
    return y_pred, calcCenters(y_pred, enc)

def calcCentersNew(y_pred, enc):
    centers = [[] for i in range(max(y_pred)+1)]
    clusters = [[]for i in range(max(y_pred)+1)]
    for i in range(len(enc)):
        if(y_pred[i]>= 0 ):
            clusters[y_pred[i]].append(enc.loc[i,:].to_dict())
            clusters.append(enc[i,:].to_dict())
    for i,_ in enumerate(clusters):
        cluster = pd.DataFrame.from_records(clusters[i])
        centers[i] = cluster.mean(axis=0).to_dict()
    return centers

def calcCenters(y_pred, enc):
    centers = [[] for i in range(max(y_pred)+1)]
    for v in range(max(y_pred)+1) :
        cluster = []
        indexPos = [ i for i in range(len(y_pred)) if y_pred[i] == v ]
        for i in indexPos :
            cluster.append(enc[i])
        centers[v] = np.mean(cluster, axis=0)
    return centers

def groupSessions (y_pred,enc):
    clusters = [[]for i in range(max(y_pred)+1)]
    for i in range(len(enc)):
        if(y_pred[i]>= 0 ):
            clusters[y_pred[i]].append(enc.loc[i,:].to_dict())
            clusters.append(enc[i,:].to_dict())
    for i,_ in enumerate(clusters):
        cluster = pd.DataFrame.from_records(clusters[i])
    return clusters


def printCluster(centers, distinct):
    indexes = np.argsort([c[range(len(distinct))] for c in centers], axis = 1)
    for i,ind in enumerate(indexes):
        for j in ind[-1:]:
                    print("cluster",i," : ",distinct[j], centers[i][j])

def checkCluster(distinct, labels, sessions):
    '''
    for i,l in enumerate(labels):
        print("label : ",l)
        for e in sessions[i].events:
            if sessions[i].encoded[distinct.index(e.name)]!= 0 :
                print(e.name,sessions[i].encoded[distinct.index(e.name)])
    '''
    for s in sessions:
        for i,e in enumerate(s.encoded):
            if e >1:
                print(distinct[i])

def elbow (X, nfrom, nto,imgpath):
    sse = []
    for k in range(nfrom, nto+1):
        kmeans = skcl.KMeans(n_clusters=k,init = "k-means++")
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(nfrom, nto+1), sse)
    plt.grid(True)
    plt.xticks(np.arange(nfrom, nto+1, 1.0))
    plt.title('Elbow curve')
    fig.savefig(imgpath+"/elbow"+str(nfrom)+"-"+str(nto)+".png") 
    plt.close()

def multirun(num, enc, runs):
    best = skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc)
    which = -1
    for i in range(runs):
        kmeans = skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc)
        if best.inertia_ > kmeans.inertia_ :
            best = kmeans
            which = i 
    #print("run numero : ", which)
    return best

def TTestKMeans (num, enc):
    runs = 0
    results = []
    means = []
    inertias = []

    while runs < 50:
        results.append(skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc))
        inertias.append(results[runs].inertia_)
        means.append( np.mean(inertias))
        runs +=1
    plt.plot(means,label = 'media')
    plt.plot(inertias,label = 'inertia')
    plt.legend(loc="lower right", title="Legend Title", frameon=False)
    plt.savefig("./img/TTest.png")
    plt.close()
    return results[np.argmin(inertias)]

def confIntMean(a, conf=0.95):
     mean, sem, m = np.mean(a), stats.sem(a), stats.t.ppf((1+conf)/2., len(a)-1)
     return mean - m*sem, mean + m*sem

def TTestKMeans2 (num, enc):
    runs = 0
    results = []
    means = []
    inertias = []
    stop = False
    left = []
    right = []
    while stop == False:
        results.append(skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc))
        inertias.append(results[runs].inertia_)
        means.append( np.mean(inertias))
        runs +=1
        if runs>1:
            l,r = confIntMean(inertias)
            left.append(l)
            right.append(r)
        else:
            left = [means[0]]
            right = [means[0]]
        if runs>5 and abs(left[-1]-right[-1])< means[-1]/100 :
            stop = True
            print(min(inertias), np.argmin(inertias))
    left[0] = min(inertias)
    right[0] = max(inertias)
    plt.fill_between(range(runs),left,right,color= "C2", alpha= 0.2)
    plt.plot(means, 'C2',label = 'Media SSE')
    plt.plot(inertias,'C1',label = 'SSE')
    plt.plot([min(inertias) for i in inertias], 'C3--',label= "Min SSE")
    plt.annotate(round(min(inertias),3), xy=(runs/2.5,min(inertias)), xytext=(runs/2.5,min(inertias)-25))
    plt.xticks(np.arange(0, runs, step=5))
    plt.legend(loc="upper right", title="Legenda", frameon=False)
    plt.savefig("./img/TTest.png")
    plt.close()
    return results[np.argmin(inertias)]

def multirunMeans(num, enc, runs):
    best = skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc)
    for i in range(runs):
        kmeans = skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc)
        if best.inertia_ > kmeans.inertia_ :
            which = i 
    return which

def calculate_kn_distance(X,k):
    kn_distance = []
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            eucl_dist.append(distance(X[i],X[j]))
        eucl_dist.sort()
        kn_distance.append(round(eucl_dist[k],2))
    return kn_distance

def epsEstimate (enc,imgpath):
    enc = enc.values.tolist()
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(enc)
    _,indices = nbrs.kneighbors(enc)
    eps_dist = []
    for i,e in enumerate(enc): 
        eps_dist.append(distance(e,enc[indices[i][-1]]))
    eps_dist.sort()
    
    tot = len(eps_dist)
    fig,ax1 = plt.subplots(figsize=(10,5))
    _, bins, _ = ax1.hist(eps_dist[:-math.floor(tot/5)],bins= 50)
    perc = [0 for i in bins]
    for i,v in enumerate(bins):
        for e in eps_dist:
            if e<v:
                perc[i]+=1
    for i,_ in enumerate(perc):
        perc[i] = (perc[i]/tot)*100
    ax1.set_xticks(bins)
    ax1.xaxis.set_tick_params(rotation=90)
    ax1.set_title("Eps Estimate ")
    ax1.set_ylabel('n')
    ax1.set_xlabel('Eps')
    ax2 = ax1.twinx()
    ax2.plot(bins,perc,color= 'red')
    ax2.set_yticks(np.linspace(0,100,num = 11))
    ax2.set_ylabel('percentage')
    ax1.grid(color= 'C0',alpha = 0.5)
    ax2.grid(color= 'red',alpha = 0.5)
    ax2.hlines(50,xmin=min(bins),xmax = max(bins),linestyle = 'dashed',color = 'k')
    ax2.tick_params(axis='y', colors='red')
    ax1.tick_params(axis='y', colors='C0')
    fig.tight_layout()
    fig.savefig(os.path.join(imgpath,"epsEstimate.png"))
    fig.clear()

def minPointsEstimate (enc, eps, imgpath):
    tree = BallTree(np.array(enc))
    allNgbr = []
    allNgbr.append(tree.query_radius(enc, eps, count_only=True))
    _, bins, _ = plt.hist (allNgbr, bins = 45)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(bins, rotation = 90 )
    plt.title("MinPts Estimate "+encoding)
    plt.ylabel('Number of sessions')
    plt.xlabel('Number of neighbors')
    plt.tight_layout()
    plt.savefig(os.path.join(imgpath,"minptsEstimate.png"))
    plt.close()

def findRuns (enc, value, clusters):
    num = []
    i = 1
    while i <= value+1:
        nsum = 0
        for j in range(5):
            print(nsum)
            nsum += multirunMeans(clusters,enc,i)
        num.append(nsum/5)
        i += 10
    print(num)
    plt.plot(np.linspace(1,value+1),num)
    plt.savefig(os.path.join(imgpath,"provaKmeans.png"))
    plt.close()

def completed (lista, index, name):
    for i in lista[index+1:]:
        try:
            if i.name == name and i.state == 'complete':
                return True
        except:
            if i['concept:name'] == name and i['lifecycle:transition'] == 'complete':
                return True
    return False

def linearEstimator(encEvents,events,sessions,distinct):
    start = time.time()
    withNan = {e:[] for e in events}
    noNan = {e:[] for e in events}
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    for e in events:
        for i,j in encEvents.iterrows():
            if str(j[e]) == 'nan':
                timestamp = None
                if e in distinct:
                    for es in sessions[i].events:
                        if es.name == e:
                            timestamp = es.timestamp
                else:
                    timestamp = sessions[i].events[0].timestamp
                j['weekday'] = int(timestamp.strftime('%w'))
                j['day'] = (timestamp.replace(tzinfo=None) - datetime(1970,1,1)).days
                j['time'] = (timestamp.replace(tzinfo=None)-midnight).seconds
                withNan[e].append(j)
            elif j[e] != np.nan:
                timestamp = None
                if e in distinct:
                    for es in sessions[i].events:
                        if es.name == e:
                            timestamp = es.timestamp
                else:
                    timestamp = sessions[i].events[0].timestamp
                if timestamp != None:
                    j['weekday'] = int(timestamp.strftime('%w'))
                    j['day'] = (timestamp.replace(tzinfo=None) - datetime(1970,1,1)).days
                    j['time'] = (timestamp.replace(tzinfo=None)-midnight).seconds
                else:
                    j['weekday'] = -1
                    j['day'] = 0
                    j['time'] = 0
                noNan[e].append(j)
        if withNan[e] != []:
            df = pd.DataFrame(noNan[e])
            imputer = sk.impute.SimpleImputer(strategy='mean', missing_values=np.nan)
            y = df[[e]]
            y_train = y[:-30]
            y_test = y[-30:]
            x = df.drop([e],axis =1)
            cols = x.columns
            x = pd.DataFrame(data = imputer.fit_transform(x), columns =cols)
            x_train =x[:-30]
            x_test = x[-30:]
            regressor = sk.linear_model.LinearRegression().fit(x_train,y_train)
            toPred = pd.DataFrame(withNan[e]).drop([e],axis=1)
            cols = toPred.columns
            toPred = pd.DataFrame(data = imputer.transform(toPred), columns =cols)
            new = regressor.predict(toPred)
            y_pred =  regressor.predict(x_test)
            error = sk.metrics.mean_squared_error(y_test,y_pred,squared = False)
            print('test',y_test)
            print('pred',y_pred)
            print('RMSE: %.2f' % error)
            flat = new
            dfNew = pd.DataFrame(withNan[e])
            dfNew[e] = flat
            encEvents = encEvents.combine_first(dfNew.drop(['time','day','weekday'],axis=1))
    encEvents.fillna(0)
    print('Estimation Time :', time.time()-start)
    return encEvents
    
def started (event, case, id):
    ind = case.index(event)
    starter = next((e for e in case[id+1:ind] if e['lifecycle:transition']== 'start' and e['concept:name']==event['concept:name']),None)
    if starter != None:
        return True
    else:
        return False

def safediv (attr,freq):
    if isinstance(attr, float):
        return attr/freq
    else:
        return attr

class Event :
    def __init__(self,event=None,attrNames=[]):
        if event == None:
            self.name = None
        else:
            if "activityNameEN" in event:
                self.name = event["activityNameEN"]
            else :
                self.name = event["concept:name"]
            self.attributes = getAttributes(event,attrNames)
            try :
                self.timestamp = event["time:timestamp"]
            except:
                self.timestamp = datetime.now()
            try: 
                self.state = event["lifecycle:transition"]
            except:
                self.state = 'complete'

    def create(self,features,attributes):
        self.name = features['name']
        self.ID = features['case']
        self.timestamp = features['timestamp']
        self.state = features['state']
        self.cluster = features['cluster']
        self.attributes = attributes

    def toDf(self,attrNames,case= None):
        attr = []
        if self.attributes != []:
            if type(self.attributes) is list:
                attr = [a for a in self.attributes]
            elif type(self.attributes) is dict:
                attr = [self.attributes[a] for a in self.attributes]
                
        if case != None:
            self.ID = case
        try :
            data = [self.name,self.ID,self.timestamp,self.state,self.cluster]
            column = ['concept:name','case','time:timestamp','lifecycle:transition','cluster']
        except:
            data = [self.name,self.ID,self.timestamp,self.state]
            column = ['concept:name','case','time:timestamp','lifecycle:transition']
        data.extend(attr)
        
        if type(self.attributes) is list:
            column.extend([attrNames[i] for i,a in enumerate(attrNames)])
        if type(self.attributes) is dict:
            column.extend([attrNames[i] for i,a in enumerate(attrNames)])
        df = pd.DataFrame([data],columns = column)
        return df
        
    def __str__(self):
        try:
            if self.attributes != []:
                strAttr = ','.join([str(a) for a in self.attributes])
                return "%s,%s,%s,%s,%s"%(self.name,self.ID, str(self.timestamp),self.state,strAttr)
            else:
                return "%s,%s,%s,%s"%(self.name,self.ID, str(self.timestamp),self.state)
        except:
            return self.name
    def __repr__(self):
        return "<Event : concept:name = %s, lifecycle:transition = %s, time:timestamp = %s>"%(self.name,self.state, str(self.timestamp))
    
    def attribute (self, name):
        return self.attributes[name].value

    def encodeAttributes (self,attrValues):
        encoded = []
        for name,attr in self.attributes.items():
            oneHot = [0 for i in range(max(attrValues[name].values())+1)]
            oneHot[attrValues[name][attr.value]] = 0.5
            encoded += oneHot
        return np.array(encoded)

class Attribute :
    def __init__(self,name,value) :
        self.name = name
        self.value = value

    def __str__ (self):
        return  "%s"%(self.value)
    def __repr__ (self):
        return  "%s"%(self.value)
    
    def encode (self,attrValues):
        index = attrValues[self.name].index(self.value)
        for k,v in attrValues.items():
            if self.name == k :
                l = [0 for j in v]
                l[index] = 1
        return l

class Session :
    def __init__ (self,caseid, events = []):
        self.ID = caseid
        self.events = events
        self.distinct = findDistinct(self.events)

    def addEvent (self, event):
        self.events.append(event)
        self.distinct = findDistinct(self.events)
    
    def timeEncoding (self,events):
        self.encoded = [0 for i in range(len(events))]
        self.frequency = [0 for i in range(len(events))]
        self.mode = "time"
        duration = [0 for i in self.distinct]
        checked = []
        for i,e in enumerate(self.events):
            self.frequency[events.index(e.name)] +=1
            index = self.distinct.index(e.name)
            if e.state == 'start':
                event_complete = next((n for n in self.events[i:] if n.state == 'complete' and n.name == e.name),None)
                if event_complete != None:
                    checked.append(i)
                    duration[index] = duration[index] + (event_complete.timestamp - e.timestamp).total_seconds()
                elif i+1<len(self.events):
                    duration[index] = duration[index] + (self.events[i+1].timestamp- e.timestamp).total_seconds()
            elif e.state == 'complete' and i not in checked:
                if i+1<len(self.events):
                    duration[index] = duration[index] + (self.events[i+1].timestamp- e.timestamp).total_seconds()
        for i,d in enumerate(duration):
            self.encoded[events.index(self.distinct[i])] = d

    def timeEncodingNew (self,eventsNames):
        self.encoded = {i:0 for i in eventsNames}
        checked = []
        for i,e in enumerate(self.events):
            if e.state == 'start':
                if i+1<(len(self.events)) and self.events[i+1].state == 'complete' and self.events[i+1].name == e.name:
                    event_complete = self.events[i+1]
                else:
                    event_complete = None
                if event_complete != None:
                    checked.append(i+1)
                    self.encoded[e.name] += (event_complete.timestamp - e.timestamp).total_seconds()
                elif i+1<len(self.events):
                    if self.encoded[e.name] == np.nan:
                        self.encoded[e.name] = 0 
                    self.encoded[e.name] += abs((self.events[i+1].timestamp- e.timestamp).total_seconds())
                elif self.encoded[e.name] == 0:
                    self.encoded[e.name] = np.nan
            elif e.state == 'complete' and i not in checked:
                if i>0:
                    if self.encoded[e.name] == np.nan:
                        self.encoded[e.name] = 0 
                    self.encoded[e.name] += abs((self.events[i-1].timestamp-e.timestamp).total_seconds())
                elif self.encoded[e.name] == 0:
                    self.encoded[e.name] = np.nan

    def frequencyEncoding(self,events):
        self.encoded = [0 for i in range(len(events))]
        self.mode = "freq"
        for i,e in enumerate(self.events):
            #if i > 0 and ( self.events[i-1].timestamp == e.timestamp and self.events[i-1].name == e.name):
            self.encoded[events.index(e.name)] += 1
        self.encoded = normalize(self.encoded)

    def freqEncodingNew(self,eventsNames):
        self.encoded = {i:0 for i in eventsNames}
        checked = []
        for i,e in enumerate(self.events):
            if e.state == 'start':
                self.encoded[e.name] +=1
                event_complete = next((n for n in self.events[i:] if n.state == 'complete' and n.name == e.name),None)
                if event_complete != None:
                    checked.append(i)
            elif e.state == 'complete' and i not in checked:
                self.encoded[e.name] +=1

    def attributesEncoding(self, attrValues):
        attrEncoded = []
        for j in attrValues.values():
            attrEncoded += [0 for i in range(max(j.values())+1)]
        for i in self.events:
            attrEncoded = np.array(attrEncoded)+i.encodeAttributes(attrValues)
        self.attrEncoded  = normalize(attrEncoded.tolist(),factor = 2)

    def addAttributesMean(self,attrNames):
        tot = [0.0 for _ in attrNames]
        num = [0 for _ in attrNames]
        for e in self.events:
            for i,a in enumerate(attrNames):
                if e.attribute(a) != None:
                    tot[i] += float(e.attribute(a))
                    num[i] += 1
        self.encoded.extend(list(np.divide(tot,num)))

    def addAttributesPrePost(self,attrNames):
        for a in attrNames:
            if self.events[0].attribute(a) != None and self.events[-1].attribute(a) != None:
                self.encoded.extend([float(self.events[0].attribute(a)), float(self.events[-1].attribute(a))])
            else:
                pre = next((e.attribute(a) for e in self.events if e.attribute(a) == None), 0)
                post = next((e.attribute(a) for e in reversed(self.events) if e.attribute(a) == None), 0)
                if pre != None:
                    pre = float(pre)
                else:
                    pre = float(0)
                if post != None:
                    post = float(post)
                else:
                    post = float(0)
                self.encoded.extend([pre,post])

    def addAttributesPrePostDF(self,attrNames):
        self.attrEncoded = {a:None for a in attrNames}
        for a in attrNames:
            if self.events[0].attribute(a) != None and self.events[-1].attribute(a) != None:
                self.attrEncoded['initial:'+a] = float(self.events[0].attribute(a))
                self.attrEncoded['final:'+a] = float(self.events[-1].attribute(a))
            else:
                pre = next((e.attribute(a) for e in self.events if e.attribute(a) == None), 0)
                post = next((e.attribute(a) for e in reversed(self.events) if e.attribute(a) == None), 0)
                if pre != None:
                    pre = float(pre)
                else:
                    pre = float(0)
                if post != None:
                    post = float(post)
                else:
                    post = float(0)
                self.attrEncoded['initial:'+a] = pre
                self.attrEncoded['final:'+a] = post

    def addAttributesMeanDF(self,attrNames):
        tot = {i:0.0 for i in attrNames}
        num = {i:0 for i in attrNames}
        for e in self.events:
            for a in attrNames:
                if e.attribute(a) != None:
                    tot[a] += float(e.attribute(a))
                    num[a] += 1
        self.attrEncoded = {k:float(tot[k]/num[k]) if num[k] >0 else np.nan for k in num }
            
    def joinEncoding(self):
        self.encoded = self.encoded + self.attrEncoded

    def abstract(self,center, cluster, distinctEvent,attrNames):
        ind = list(np.flip(np.argsort(center[:len(distinctEvent)])[-1:]))
        name = concatName(distinctEvent,ind)
        eventS = {
            "name" : name+str(cluster),
            "case" :self.ID,
            "timestamp" : self.events[0].timestamp,
            "state" :"start",
            "cluster": cluster
        }
        eventC = {
            "name" : name+str(cluster),
            "case" :self.ID,
            "timestamp" : self.events[-1].timestamp,
            "state" :"complete",
            "cluster": cluster
        }
        attributes = []
        if len(center) > len(distinctEvent):
            fromcenter = center[len(distinctEvent):]
            for i,an in enumerate(attrNames):
                attributes.append(Attribute(an,fromcenter[i]))
        else:
            attrV = {a:0 for a in attrNames}
            freq = {a:0 for a in attrNames}
            for e in self.events:
                for a in attrNames:
                    if e.attribute(a) != None:
                        #print(attrV, e.attribute(a))
                        try:
                            attrV[a] += float(e.attribute(a))
                            freq[a] +=1
                        except:
                            attrV[a] = e.attribute(a)
            attributes = [Attribute(k,safediv(attrV[k],freq[k])) if freq[k] > 0 else Attribute(k,None) for k in attrV]
        abstractedS = Event()
        abstractedC = Event()
        abstractedS.create(eventS,attributes)
        abstractedC.create(eventC,attributes)
        return [abstractedS.toDf(attrNames),abstractedC.toDf(attrNames)]
    
    def convert (self,center,cluster,distinctEvent,attrNames):
        es = self.abstract(center,cluster,distinctEvent,attrNames)
        strForm = str(es)
        return strForm

    def convertSession (self,center,cluster,distinctEvent,attrNames):
        frames = self.abstract(center,cluster,distinctEvent,attrNames)
        return pd.concat(frames,ignore_index = True)

    def export(self,attrNames,case):
        toDF = []
        for e in self.events:
            toDF.append(e.toDf(attrNames, case = case))
        return toDF

class Log :
    def __init__(self, name, path, fileName=None,threshold=10,attrNames=[],noTrace = None,toSession = True):
        self.name = name
        fileName = name if not fileName else fileName
        start = time.time()
        self.log = importLog(path, fileName)
        print("Importing Time:",time.time()-start)
        if noTrace != None:
            self.tracefy(limit = noTrace)
        if attrNames== [] and toSession == True:
            attrNames = pm.get_attributes(self.log)
            #print(attrNames)
            toRemove = ['org:resource','time:timestamp','(case)_variant-index','name','position','(case)_creator','concept:name','(case)_variant']
            for i in toRemove:
                if i in attrNames:
                    attrNames.remove(i)
        if toSession == True:
            print('cerco sessioni')
            self.sessions = self.findSessionsNew(threshold,attrNames)

    def thresholdView (self,imgpath):
        difference = []
        for caseid,case in enumerate(self.log):
            added = []
            for event_id, event in enumerate(case):
                if event['lifecycle:transition'] == 'start' and event_id>0:
                    difference.append ((event["time:timestamp"]-next((e['time:timestamp'] for e in reversed(case[:event_id]) if e['lifecycle:transition'] == 'start'),event["time:timestamp"])).total_seconds())
                    ec = next((e for e in case[event_id:] if e['concept:name'] == event['concept:name'] and e['lifecycle:transition'] == 'complete'),None)
                    if ec != None and not started(ec, case, event_id):
                        added.append(case.index(ec))
                elif event['lifecycle:transition'] == 'complete' and event_id>0 and event not in added:
                    difference.append((event["time:timestamp"]-next((e['time:timestamp'] for e in reversed(case[:event_id]) if e['lifecycle:transition'] == 'start'),case[-1]['time:timestamp'])).total_seconds())
        if difference != []:
            mean = np.mean([d for d in difference if d>0])
            median = st.median([d for d in difference if d>0])
            print('mean',mean)
            plt.plot(difference,marker='o',linewidth = 0,zorder=1)
            plt.hlines(mean,xmin = 0, xmax = len(difference),color='red',linewidth=2,zorder = 2,label= 'Mean value = '+str(mean/60)+' min')
            plt.hlines(median,xmin = 0, xmax = len(difference),color='orange',linewidth=2,zorder = 3,label= 'Median value = '+str(median/60)+' min')
            plt.legend()
            plt.savefig(os.path.join(imgpath,'thresholdView.png'))
            plt.close()
            print('THRESHOLD VARIANCE' , np.std(np.divide(difference,[60 for i in difference])))
    
    def tracefy (self,limit):
        log = self.log[0]
        traced = []
        trace = []
        weekend = []
        for event_id, event in enumerate(log):
            if event_id == 0 or trace == []:
                trace.append(event)
            elif event['lifecycle:transition'] == 'start' and (trace[0]["time:timestamp"].hour <limit and event["time:timestamp"].   day == trace[0]["time:timestamp"].day and event["time:timestamp"].hour <limit) or (trace[0]["time:timestamp"].hour >= limit and ((
                    (event["time:timestamp"].day == (trace[0]["time:timestamp"].day)+1) and event["time:timestamp"].hour <limit) or 
                    (event["time:timestamp"].day == trace[0]["time:timestamp"].day and event["time:timestamp"].hour>=limit))):
                trace.append(event)
            elif event['lifecycle:transition'] == 'complete':
                starter = next((i for i,e in enumerate(reversed(trace)) if e['lifecycle:transition'] == 'start' and e['concept:name'] == event['concept:name']),None)
                if starter != None and not completed(trace,starter,event['concept:name']):
                        trace.append(event)
                elif (trace[0]["time:timestamp"].hour <limit and event["time:timestamp"].   day == trace[0]["time:timestamp"].day and event["time:timestamp"].hour <limit) or (trace[0]["time:timestamp"].hour >= limit and (((event["time:timestamp"].day == (trace[0]["time:timestamp"].day)+1) and event["time:timestamp"].hour <limit) or (event["time:timestamp"].day == trace[0]["time:timestamp"].day and event["time:timestamp"].hour>=limit))):
                    trace.append(event)
                else:
                    traced.append(trace)
                    trace = [event]
            else:
                traced.append( trace)
                trace = [event]
        self.log = traced

    def findSessionsNew(self,threshold,attrNames):
        start = time.time()
        self.attrNames = attrNames
        sessions = []
        for caseid,case in enumerate(self.log):
            added = []
            for event_id, event in enumerate(case):
                if event_id == 0 or session.events == []:
                    session = Session(caseid=caseid,events = [Event(event,attrNames)])
                elif event['lifecycle:transition'] == 'start':
                    if event["time:timestamp"]-next((e.timestamp for e in reversed(session.events) if e.state == 'start'),session.events[-1].timestamp)<=timedelta(minutes=threshold):
                        session.addEvent(Event(event,attrNames))
                        ec = next((e for e in case[event_id:] if e['concept:name'] == event['concept:name'] and e['lifecycle:transition'] == 'complete'),None)
                        if ec != None and not started(ec, case, event_id) and (ec['time:timestamp'] - event['time:timestamp'])<= timedelta(hours=11):
                            session.addEvent(Event(ec,attrNames))
                            added.append(ec)
                    else:
                        sessions.append(session)
                        session = Session(caseid=caseid, events = [Event(event,attrNames)])
                elif event['lifecycle:transition'] == 'complete':
                    if  event not in added:
                        if event["time:timestamp"]-next((e.timestamp for e in reversed(session.events) if e.state == 'start'),session.events[-1].timestamp)<=timedelta(minutes=threshold):
                            session.addEvent(Event(event,attrNames))
                        else:
                            sessions.append(session)
                            session = Session(caseid=caseid, events = [Event(event,attrNames)])
                '''
                else :
                    if session.events!=[]:
                        sessions.append(session)
                    session = Session(caseid=caseid, events = [Event(event,attrNames)])
                '''
            if sessions != [] and session != sessions[-1] and session.events!=[]:
                sessions.append(session)
                session = Session(caseid=caseid)
        distinct = []
        for i in sessions:
            distinct = list(set(distinct) | set(i.distinct))
        distinct.sort()
        self.distinct = distinct
        print("Sessioning Time:",time.time()-start)
        return sessions

    def encode (self, encoding,norm = None,useAttributes = False):
        start = time.time()
        encDf = pd.DataFrame([], columns = self.distinct)
        if encoding == "time":
            for i in range(len(self.sessions)):
                self.sessions[i].timeEncodingNew(self.distinct)
                encDf = encDf.append(pd.Series(self.sessions[i].encoded, index = encDf.columns),ignore_index= True)
                if useAttributes:
                    self.sessions[i].addAttributesMeanDF(self.attrNames)
                    for j in self.attrNames:
                        encDf.loc[i,j] = self.sessions[i].attrEncoded[j]
        if encoding == 'freq':
            for i in range(len(self.sessions)):
                self.sessions[i].freqEncodingNew(self.distinct)
                encDf = encDf.append(pd.Series(self.sessions[i].encoded, index = encDf.columns),ignore_index= True)
                if useAttributes:
                    self.sessions[i].addAttributesMeanDF(self.attrNames)
                    for j in self.attrNames:
                        encDf.loc[i,j] = self.sessions[i].attrEncoded[j]
        if encoding == 'time' or useAttributes:
            encDf = linearEstimator(encDf,encDf.columns,self.sessions,self.distinct)
        onlyEvents = encDf.loc[:,self.distinct]
        if useAttributes:
            onlyAttr = encDf.loc[:,self.attrNames]
            onlyAttr = self.normalizeAttrNew(onlyAttr,self.attrNames)
            encDf.loc[:,self.attrNames] = onlyAttr
        if norm == 'session':
            onlyEvents = onlyEvents.div(onlyEvents.sum(axis=1),axis=0).replace(np.nan,0)
        elif norm == 'absolute':
            onlyEvents = self.normalizeEvents(onlyEvents)
        encDf.loc[:,self.distinct] = onlyEvents
        self.encodedLog = encDf
        print("Encoding Time:",time.time()-start)

    def normalizeAttrNew(self,attributes,newNames):
        shift = {i: abs(math.floor(np.min(attributes.loc[:,[i]].values.tolist()))) if np.min(attributes.loc[:,[i]].values.tolist()) <0 else 0  for i in self.attrNames}
        for i in newNames:
            attributes.loc[:,i] = attributes.loc[:,i]+shift[i]
        lower = {i: np.min(attributes.loc[:,[i]].values.tolist()) for i in self.attrNames}
        higher = {i: np.max(attributes.loc[:,[i]].values.tolist()) for i in self.attrNames}
        for i in newNames:
            attributes.loc[:,i] = (attributes.loc[:,i]-lower[i]).div(higher[i]-lower[i])
        return attributes

    def normalizeEvents(self,events):
        lower = {i: np.min(events.loc[:,[i]].values.tolist()) for i in self.distinct}
        higher = {i: np.max(events.loc[:,[i]].values.tolist()) for i in self.distinct}
        for i in self.distinct:
            events.loc[:,i] = (events.loc[:,i]-lower[i]).div(higher[i]-lower[i])
        return events.fillna(0)
    
    def findFrequency(self,max):
        absFreq = np.array([0 for i in self.distinct])
        for s in self.sessions:
            absFreq = np.sum([absFreq,s.frequency], axis=0)
        return absFreq.tolist()

    def attrCipher(self):
        values = {i:{} for i in self.attrNames}
        unique = {i:0 for i in self.attrNames}
        for s in self.sessions:
            for e in s.events:
                for a in e.attributes.values():
                    if a.value not in values[a.name].keys():
                        values[a.name][a.value] = unique[a.name]
                        unique[a.name] +=1
        return values

    def cluster (self,params = {"alg":"KMeans","num":10}):
        start = time.time()
        encodedLog = self.encodedLog.values.tolist()
        if params["alg"].lower() == "kmeans":
            if not "runs" in params:
                params["runs"] = 0
            cluster = TTestKMeans2(params["num"],encodedLog)
            print("SSE : ", cluster.inertia_)
            print("Clustering Time:",time.time()-start)
            return cluster.predict(encodedLog),cluster.cluster_centers_
        elif params["alg"].lower() == "dbscan":
            cluster = skcl.DBSCAN(min_samples=params["minsamples"], eps = params["eps"]).fit(encodedLog)
            y_pred = cluster.labels_
            centers = calcCenters(y_pred, encodedLog)
            print("Clustering Time:",time.time()-start)
            if "assignNoisy" in params and params["assignNoisy"] == True:
                y_pred, centers = assignNoisyPoints(y_pred,encodedLog,centers)
            return y_pred,centers
    
    def exportSubP (self,y_pred,center,name,encoding,alg):
        path = os.pardir+'/outputLog/'+encoding+'/'+alg
        try:
            os.makedirs(path,exist_ok=True)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)
            frames = [[] for i in range(max(y_pred)+1)]
            for i,s in enumerate(self.sessions):
                frames[y_pred[i]].extend(s.export(self.attrNames,i))
            for i in range(max(y_pred)+1):
                ind = list(np.flip(np.argsort(centers[i][:len(self.distinct)])[-1:]))
                subP = concatName(self.distinct,ind)
                newFile = name+str(i)+subP+'.xes'
                log = pd.concat(frames[i],ignore_index=True)
                log = pm.format_dataframe(log,case_id='case',activity_key='concept:name',timestamp_key='time:timestamp')
                log = pm.convert_to_event_log(log)
                pm.write_xes(log,os.path.join(path,newFile))
            print("Sessions exported")


    def convertLog(self,centers, y_pred,name,encoding,alg,datapath,exportSes = False):
        start = time.time()
        frames = []
        log = pd.DataFrame()
        for i,s in enumerate(self.sessions):
            abstracted = s.convertSession(centers[y_pred[i]],y_pred[i], self.distinct,self.attrNames)
            frames.append(abstracted)
        log = pd.concat(frames,ignore_index=True)
        log = pm.format_dataframe(log,case_id='case',activity_key='concept:name',timestamp_key='time:timestamp')
        num = math.ceil(len(log)*0.7)
        
        log1 = log[:num]
        log2 = log[num:]

        log = pm.convert_to_event_log(log)
        log1 = pm.convert_to_event_log(log1)
        log2 = pm.convert_to_event_log(log2)

        pm.write_xes(log1,os.path.join(datapath,name+"train.xes"))
        pm.write_xes(log2,os.path.join(datapath,name+"test.xes"))
        pm.write_xes(log,os.path.join(datapath,name+".xes"))

        if exportSes:
            self.exportSubP(y_pred,centers,name,encoding,alg)
        print("Convertion Time:",time.time()-start)

    def renameTasks (self,names,datapath):
        if names != {}:
            try:
                for ic,case in enumerate(self.log):
                    for ie,event in enumerate(case):
                            if event['cluster'] in names:
                                self.log[ic][ie]['concept:name'] = names[event['cluster']]
                pm.write_xes(self.log,os.path.join(datapath,self.name+".xes"))
            except:
                print('Event-log not suitable for renaming')
        else:
            print('No name given')

            
    def betterPlotting (self,centers, y_pred, path, params,mode = "linear"):
        distinct = self.distinct
        attrValues = self.attrNames
        if (len(centers[0])> len(distinct)):
            attr = [c[len(distinct):] for c in centers]
            attrNames = attrValues
        else:
            attr = []
            attrNames = []
        centers = np.array([c[range(len(distinct))] for c in centers])
        
        #Normalizzazione solo per plotting
        if any(i>1 for c in centers for i in c):
            for i,c in enumerate(centers):
                lower =  min(c)
                if lower <0 :
                    lower = abs(lower)
                    centers[i] = [j+lower for j in c]
            centers = centers/centers.sum(axis=1,keepdims = True)
        newCenters= []
        newDistinct = []
        for i,e in enumerate(distinct):
            drop = True
            for c in centers:
                if c[i] >= 0.01 :
                    drop = False
            if not drop:
                newDistinct.append(e)
        for i,c in enumerate(centers):
            cn = []
            for e in newDistinct:
                cn.append(c[distinct.index(e)])
            if attr != [] :
                cn = [*cn, *attr[i]]
            newCenters.append(cn)
        if attr != []:
            columns = newDistinct + attrNames
        else:
            columns = newDistinct
        df1 = pd.DataFrame(newCenters,index=range(max(y_pred)+1),columns =columns)
        logmin = 0.001
        fig, ax = plt.subplots()
        fig.set_size_inches((len(columns),len(newCenters)))
        if mode == "linear":
            sns.heatmap(df1, cmap="YlOrRd", linewidths=.5,xticklabels=True, yticklabels= True, ax = ax)
        else:
            sns.heatmap(df1, cmap="YlOrRd", linewidths=.5,norm =LogNorm(), vmin= max(centers.min().min(),logmin),xticklabels=True, yticklabels= True, ax= ax)
        if attr != []:
            ax.vlines([len(newDistinct)], *ax.get_ylim())
        ax.set_title(params)
        fig.savefig(path+".png",bbox_inches='tight') 
        ax.clear()



logpath = "../inputLog/IOT"
datapath = "../outputLog"
imgpath = "./img"

logname = "Log5processato"
encoding = "freq"
print(logname,encoding)
attr = []
params = {
    "alg":"kmeans".upper(),
    "minsamples":52,
    "eps":0.02,
    "num" :20,
    "assignNoisy" : True
}

mylog = Log(name = logname,path =logpath,attrNames=attr,threshold=3,noTrace=19)
mylog.encode(encoding,norm ='session',useAttributes = False)

print("Sessions number: ",len(mylog.sessions))


epsEstimate(mylog.encodedLog,imgpath)
minPointsEstimate(mylog.encodedLog, 0.07,imgpath)

#elbow(mylog.encodedLog,10,40,imgpath)

y_pred,centers = mylog.cluster(params)

attrString = str(attr).replace("[","").replace("]","").replace("'","")

mylog.betterPlotting(centers,y_pred,os.path.join(imgpath,logname+encoding+attrString+params["alg"]),params)


printCluster(centers, mylog.distinct)
mylog.convertLog(centers, y_pred,logname+encoding+attrString+params["alg"],encoding,params['alg'],datapath,exportSes = True)

#renameLog = Log(name = 'Log5NuovotimeKMEANS' ,path = datapath ,attrNames=attr,toSession=False)

#renameLog.renameTasks({0:'OP'},datapath)
